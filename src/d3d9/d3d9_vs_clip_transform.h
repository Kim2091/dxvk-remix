/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once

#include <cstdint>
#include <vector>

#include "../util/util_matrix.h"
#include "../util/util_vector.h"

namespace dxvk {

  // Recovers the object->clip transform a D3D9 vertex shader actually applies, from its
  // bytecode plus the live float constant registers.
  //
  // Why this exists: camera acquisition in NGX passthrough mode has to work on games whose
  // shaders declare no recognisable camera symbol (unknown naming conventions) or no constant
  // table at all (stripped at build time), and on games that never upload the camera as one
  // whole 4x4 (a 3-register affine world->view plus a handful of packed projection scalars is
  // a common compact form). Names cannot answer those; the arithmetic can. Whatever constants
  // combine to produce clip-space position ARE the transform, however they are spelled and
  // however they are split up.
  //
  // The analysis is a plain abstract interpretation over the DXSO instruction stream in which
  // every value is an affine form over the four components of the shader's POSITION input:
  //
  //     value = k[0]*pos.x + k[1]*pos.y + k[2]*pos.z + k[3]*pos.w + k[4]
  //
  // Constant registers are affine forms with only k[4] set, the position input contributes the
  // identity forms, and anything else (other vertex attributes, flow control, unmodelled
  // opcodes) is Unknown and poisons whatever reads it. If clip position survives as an affine
  // form then the shader's transform is exactly a 4x4 matrix and it is recovered exactly; if it
  // does not, the analysis says so rather than guessing.
  //
  // The structure (which opcodes, which registers, what is modelable) depends only on the
  // bytecode, so it is resolved once per shader; only the arithmetic needs the live constants,
  // which is what evaluateVsClipTransform runs per draw.
  //
  // NOTE: the recovered matrix is object->clip. Whether that equals world->clip depends on
  // whether the draw's object transform is identity, which the bytecode cannot say - that is
  // the caller's problem (see the clip-transform probe in D3D9Rtx, which resolves it by
  // cross-draw agreement: a world->clip yields the same frustum apex for every draw in a
  // frame, an object->clip does not).

  enum class VsClipTransformStatus : uint32_t {
    Ok = 0,
    NotAVertexShader,       // bytecode header is not vs_*
    DecodeFailed,           // the DXSO stream did not decode
    NoPositionInput,        // no dcl_position input register
    NoPositionOutput,       // clip position is never written
    PreTransformedPosition, // dcl_positiont: the game hands over screen space, no camera exists
    PositionNotAffine,      // clip position depends on values this analysis cannot model
    TooComplex,             // the position slice exceeds the instruction budget
  };
  const char* vsClipTransformStatusLabel(VsClipTransformStatus status);

  // Internal opcode set of the recovered program. Deliberately its own enum (not DxsoOpcode)
  // so consumers of the result do not need the DXSO headers.
  enum class VsClipOpcode : uint8_t {
    Mov = 0,
    Add,
    Sub,
    Mul,
    Mad,
    Dp3,
    Dp4,
    Dp2Add,
    Lrp,
    MatrixDp4,   // m4x4 / m4x3: dst.i = dot4(src0, const[base + i])
    MatrixDp3,   // m3x4 / m3x3 / m3x2: dst.i = dot3(src0, const[base + i])
    // Scalar operations that only ever run on values with no position dependence; folded
    // exactly at evaluation time
    Rcp,
    Rsq,
    Min,
    Max,
    Slt,
    Sge,
    Exp,
    Log,
    Frc,
    Abs,
    Pow,
    Nrm3,
    Sgn,
    Cmp,
    Cnd,
    Crs,
  };

  enum class VsClipSrcFile : uint8_t {
    Temp = 0,
    FloatConst,   // app-supplied c# register
    Literal,      // shader-declared def c#, resolved at analysis time
    Position,     // the POSITION0 input register
  };

  struct VsClipSrc {
    VsClipSrcFile file = VsClipSrcFile::Temp;
    uint8_t  swizzle = 0xE4;   // packed 2 bits per component, D3D order
    uint8_t  modifier = 0;     // DxsoRegModifier
    uint16_t index = 0;        // temp index, constant register, or index into literals
  };

  struct VsClipOp {
    VsClipOpcode opcode = VsClipOpcode::Mov;
    uint8_t  dstIsClipPosition = 0;
    uint8_t  dstIndex = 0;     // temp index when dstIsClipPosition == 0
    uint8_t  dstMask = 0xF;
    uint8_t  srcCount = 0;
    float    dstScale = 1.0f;  // instruction result shift modifier (_x2 / _d2 / ...)
    VsClipSrc src[3];
  };

  struct VsClipTransformProgram {
    VsClipTransformStatus status = VsClipTransformStatus::NotAVertexShader;

    bool ok() const {
      return status == VsClipTransformStatus::Ok;
    }

    std::vector<VsClipOp> ops;
    std::vector<Vector4> literals;

    // Float constant registers the clip position actually depends on, sorted and unique.
    // This is the engine-agnostic answer to "where is the camera in constant space": whatever
    // is in here is part of the vertex transform, and nothing outside it is.
    std::vector<uint16_t> constRegisters;
    uint32_t maxConstRegister = 0;

    // Diagnostics
    uint32_t positionInputRegister = 0;
    uint32_t instructionCount = 0;      // instructions in the decoded shader
    uint32_t sliceInstructionCount = 0; // instructions kept after the backward slice

    // Why a PositionNotAffine shader was declined, specifically for the common case of
    // relative (a0-indexed) constant addressing - c[a0 + k] reads whose exact register a static
    // analysis cannot know. This distinguishes "the analysis does not model indexed constants
    // yet" (recoverable: if the address register is loaded from a constant it is fixed for the
    // whole draw, so the read resolves at evaluation time) from "the transform genuinely varies
    // per vertex" (the address comes from vertex data). relativeAddressSourceReg is the constant
    // register feeding a0 via mova when constant-derived; relativeReadBaseReg is the base of the
    // first indexed read. Both 0xFFFF when unset.
    bool usesRelativeAddressing = false;
    bool relativeAddressConstantDerived = false;
    uint16_t relativeAddressSourceReg = 0xFFFF;
    uint16_t relativeReadBaseReg = 0xFFFF;
  };

  // Static, bytecode-only. Safe to cache per shader hash; never touches device state.
  VsClipTransformProgram analyzeVsClipTransform(const void* bytecode, size_t byteSize);

  // Executes the recovered program against live constants. Returns the object->clip transform
  // in the runtime's convention (clip = M * objectPosition, M[i] is column i), assuming the
  // position input's w component is 1 - which the D3D9 input assembler guarantees for FLOAT1/2/3
  // position formats and which every engine relies on for FLOAT4 as well.
  bool evaluateVsClipTransform(const VsClipTransformProgram& program,
                               const Vector4* floatConstants,
                               uint32_t floatConstantCount,
                               Matrix4& outObjectToClip);

}
