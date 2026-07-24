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
#include "d3d9_vs_clip_transform.h"

#include "../dxso/dxso_decoder.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace dxvk {

  const char* vsClipTransformStatusLabel(VsClipTransformStatus status) {
    switch (status) {
      case VsClipTransformStatus::Ok:                     return "ok";
      case VsClipTransformStatus::NotAVertexShader:       return "not a vertex shader";
      case VsClipTransformStatus::DecodeFailed:           return "bytecode did not decode";
      case VsClipTransformStatus::NoPositionInput:        return "no POSITION input declared";
      case VsClipTransformStatus::NoPositionOutput:       return "clip position never written";
      case VsClipTransformStatus::PreTransformedPosition: return "pre-transformed position (no camera)";
      case VsClipTransformStatus::PositionNotAffine:      return "clip position is not an affine function of the input position";
      case VsClipTransformStatus::TooComplex:             return "position slice exceeds the instruction budget";
    }
    return "unknown";
  }

  namespace {

    // Abstract value lattice: what a register component is, as a function of the shader's
    // POSITION input. Ordered so that combining two states is a max().
    enum class CompState : uint8_t {
      Unknown  = 0,  // depends on something not modelled - poisons anything reading it
      Constant = 1,  // no position dependence
      Affine   = 2,  // linear in the position input plus an offset
    };

    constexpr uint32_t kMaxTempRegisters = 32;   // DxsoMaxTempRegs
    constexpr uint32_t kMaxSliceOps      = 96;   // bounds the per-draw evaluation cost
    constexpr uint32_t kMaxConstFootprint = 64;

    struct RegStates {
      CompState c[4] = { CompState::Unknown, CompState::Unknown,
                         CompState::Unknown, CompState::Unknown };
    };

    inline CompState combine(CompState a, CompState b) {
      if (a == CompState::Unknown || b == CompState::Unknown)
        return CompState::Unknown;
      return a > b ? a : b;
    }

    // Products are only representable when at least one side carries no position dependence
    inline CompState combineProduct(CompState a, CompState b) {
      if (a == CompState::Unknown || b == CompState::Unknown)
        return CompState::Unknown;
      if (a == CompState::Affine && b == CompState::Affine)
        return CompState::Unknown;
      return a > b ? a : b;
    }

    inline uint32_t swizzleComponent(uint8_t swizzle, uint32_t index) {
      return (uint32_t(swizzle) >> (index * 2u)) & 0x3u;
    }

    // Source modifiers that stay inside the affine algebra (scale and offset only)
    inline bool modifierIsAffine(DxsoRegModifier modifier) {
      switch (modifier) {
        case DxsoRegModifier::None:
        case DxsoRegModifier::Neg:
        case DxsoRegModifier::Bias:
        case DxsoRegModifier::BiasNeg:
        case DxsoRegModifier::Sign:
        case DxsoRegModifier::SignNeg:
        case DxsoRegModifier::Comp:
        case DxsoRegModifier::X2:
        case DxsoRegModifier::X2Neg:
          return true;
        default:
          return false;
      }
    }

    // Modifiers that are only usable on values with no position dependence
    inline bool modifierIsConstantOnly(DxsoRegModifier modifier) {
      return modifier == DxsoRegModifier::Abs || modifier == DxsoRegModifier::AbsNeg;
    }

    struct OpClass {
      bool     supported = false;
      bool     linear = false;      // participates in the affine algebra
      uint32_t srcCount = 0;
      VsClipOpcode opcode = VsClipOpcode::Mov;
      uint32_t matrixRows = 0;      // matrix macros only
    };

    OpClass classifyOpcode(DxsoOpcode opcode) {
      OpClass r;
      auto set = [&r](VsClipOpcode op, uint32_t srcCount, bool linear, uint32_t rows = 0) {
        r.supported = true;
        r.opcode = op;
        r.srcCount = srcCount;
        r.linear = linear;
        r.matrixRows = rows;
      };

      switch (opcode) {
        case DxsoOpcode::Mov:    set(VsClipOpcode::Mov, 1, true); break;
        case DxsoOpcode::Add:    set(VsClipOpcode::Add, 2, true); break;
        case DxsoOpcode::Sub:    set(VsClipOpcode::Sub, 2, true); break;
        case DxsoOpcode::Mul:    set(VsClipOpcode::Mul, 2, true); break;
        case DxsoOpcode::Mad:    set(VsClipOpcode::Mad, 3, true); break;
        case DxsoOpcode::Dp3:    set(VsClipOpcode::Dp3, 2, true); break;
        case DxsoOpcode::Dp4:    set(VsClipOpcode::Dp4, 2, true); break;
        case DxsoOpcode::Dp2Add: set(VsClipOpcode::Dp2Add, 3, true); break;
        case DxsoOpcode::Lrp:    set(VsClipOpcode::Lrp, 3, true); break;
        case DxsoOpcode::M4x4:   set(VsClipOpcode::MatrixDp4, 2, true, 4); break;
        case DxsoOpcode::M4x3:   set(VsClipOpcode::MatrixDp4, 2, true, 3); break;
        case DxsoOpcode::M3x4:   set(VsClipOpcode::MatrixDp3, 2, true, 4); break;
        case DxsoOpcode::M3x3:   set(VsClipOpcode::MatrixDp3, 2, true, 3); break;
        case DxsoOpcode::M3x2:   set(VsClipOpcode::MatrixDp3, 2, true, 2); break;
        // Non-affine: exact only when nothing on the position path reaches them
        case DxsoOpcode::Rcp:    set(VsClipOpcode::Rcp, 1, false); break;
        case DxsoOpcode::Rsq:    set(VsClipOpcode::Rsq, 1, false); break;
        case DxsoOpcode::Min:    set(VsClipOpcode::Min, 2, false); break;
        case DxsoOpcode::Max:    set(VsClipOpcode::Max, 2, false); break;
        case DxsoOpcode::Slt:    set(VsClipOpcode::Slt, 2, false); break;
        case DxsoOpcode::Sge:    set(VsClipOpcode::Sge, 2, false); break;
        case DxsoOpcode::Exp:    set(VsClipOpcode::Exp, 1, false); break;
        case DxsoOpcode::ExpP:   set(VsClipOpcode::Exp, 1, false); break;
        case DxsoOpcode::Log:    set(VsClipOpcode::Log, 1, false); break;
        case DxsoOpcode::LogP:   set(VsClipOpcode::Log, 1, false); break;
        case DxsoOpcode::Frc:    set(VsClipOpcode::Frc, 1, false); break;
        case DxsoOpcode::Abs:    set(VsClipOpcode::Abs, 1, false); break;
        case DxsoOpcode::Pow:    set(VsClipOpcode::Pow, 2, false); break;
        case DxsoOpcode::Nrm:    set(VsClipOpcode::Nrm3, 1, false); break;
        case DxsoOpcode::Sgn:    set(VsClipOpcode::Sgn, 1, false); break;
        case DxsoOpcode::Cmp:    set(VsClipOpcode::Cmp, 3, false); break;
        case DxsoOpcode::Cnd:    set(VsClipOpcode::Cnd, 3, false); break;
        case DxsoOpcode::Crs:    set(VsClipOpcode::Crs, 2, false); break;
        default: break;
      }

      return r;
    }

    // Instructions after which nothing can be trusted: whether the code that follows runs at
    // all (or how many times) is a runtime property, so every later write is treated as
    // Unknown. Values already computed keep their meaning, which is why a shader that
    // transforms position before its first branch still analyses.
    bool isFlowControl(DxsoOpcode opcode) {
      switch (opcode) {
        case DxsoOpcode::Call:
        case DxsoOpcode::CallNz:
        case DxsoOpcode::Loop:
        case DxsoOpcode::EndLoop:
        case DxsoOpcode::Ret:
        case DxsoOpcode::Label:
        case DxsoOpcode::Rep:
        case DxsoOpcode::EndRep:
        case DxsoOpcode::If:
        case DxsoOpcode::Ifc:
        case DxsoOpcode::Else:
        case DxsoOpcode::EndIf:
        case DxsoOpcode::Break:
        case DxsoOpcode::BreakC:
        case DxsoOpcode::BreakP:
          return true;
        default:
          return false;
      }
    }

  }

  VsClipTransformProgram analyzeVsClipTransform(const void* bytecode, size_t byteSize) {
    VsClipTransformProgram result;

    if (bytecode == nullptr || byteSize < sizeof(uint32_t) || (byteSize % sizeof(uint32_t)) != 0) {
      result.status = VsClipTransformStatus::NotAVertexShader;
      return result;
    }

    const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode);
    const uint32_t headerToken = tokens[0];
    if ((headerToken & 0xffff0000u) != 0xfffe0000u) {
      result.status = VsClipTransformStatus::NotAVertexShader;
      return result;
    }

    const uint32_t majorVersion = (headerToken >> 8) & 0xffu;
    const uint32_t minorVersion = headerToken & 0xffu;

    std::vector<VsClipOp> ops;
    std::vector<Vector4> literals;
    // Constant registers carrying a shader-declared literal (def c#), which shadow whatever the
    // application uploaded to the same register
    std::vector<std::pair<uint32_t, uint32_t>> literalRegisters;  // register -> literal index

    RegStates temps[kMaxTempRegisters];
    RegStates clipOut;

    bool haveClipOutput = false;
    uint32_t clipOutputRegister = 0;
    bool havePositionInput = false;
    uint32_t positionInputRegister = 0;
    bool preTransformed = false;
    bool inFlowControl = false;
    uint32_t instructionCount = 0;

    // Address register (a0) provenance, tracked so a PositionNotAffine caused purely by
    // relative constant addressing can be told apart from a truly per-vertex transform. addrState
    // is the classification of the value last written to a0 (Constant = fixed per draw and thus
    // resolvable at evaluation time; Affine = built from the position input; Unknown = anything
    // else). See the diagnostic fields on VsClipTransformProgram.
    CompState addrState = CompState::Unknown;
    uint32_t addrSourceReg = 0xFFFFu;
    bool usesRelativeAddressing = false;
    bool relativeAddressConstantDerived = false;
    uint32_t relativeAddressSourceReg = 0xFFFFu;
    uint32_t relativeReadBaseReg = 0xFFFFu;

    // Pre-3.0 vertex shaders write clip position to the fixed oPos slot; 3.0 declares an
    // output register with a POSITION semantic instead
    const bool usesDeclaredOutput = majorVersion >= 3;
    if (!usesDeclaredOutput) {
      haveClipOutput = true;
      clipOutputRegister = RasterOutPosition;
    }

    try {
      DxsoProgramInfo programInfo { DxsoProgramTypes::VertexShader, minorVersion, majorVersion };
      DxsoDecodeContext decoder(programInfo);
      DxsoCodeIter iter(tokens + 1);

      while (decoder.decodeInstruction(iter)) {
        const DxsoInstructionContext& ctx = decoder.getInstructionContext();
        const DxsoOpcode opcode = ctx.instruction.opcode;
        instructionCount++;

        if (opcode == DxsoOpcode::Comment || opcode == DxsoOpcode::Nop) {
          continue;
        }

        if (opcode == DxsoOpcode::Dcl) {
          if (ctx.dst.id.type == DxsoRegisterType::Input) {
            if (ctx.dcl.semantic.usage == DxsoUsage::Position && ctx.dcl.semantic.usageIndex == 0) {
              havePositionInput = true;
              positionInputRegister = ctx.dst.id.num;
            } else if (ctx.dcl.semantic.usage == DxsoUsage::PositionT) {
              preTransformed = true;
            }
          } else if (usesDeclaredOutput && ctx.dst.id.type == DxsoRegisterType::Output &&
                     ctx.dcl.semantic.usage == DxsoUsage::Position &&
                     ctx.dcl.semantic.usageIndex == 0) {
            haveClipOutput = true;
            clipOutputRegister = ctx.dst.id.num;
          }
          continue;
        }

        if (opcode == DxsoOpcode::Def) {
          if (ctx.dst.id.type == DxsoRegisterType::Const && literals.size() < 256) {
            literalRegisters.emplace_back(ctx.dst.id.num, uint32_t(literals.size()));
            literals.emplace_back(ctx.def.float32[0], ctx.def.float32[1],
                                  ctx.def.float32[2], ctx.def.float32[3]);
          }
          continue;
        }

        if (opcode == DxsoOpcode::DefI || opcode == DxsoOpcode::DefB) {
          continue;
        }

        if (isFlowControl(opcode)) {
          inFlowControl = true;
          continue;
        }

        // Address register write (mova a0, ... / mov a0, ...): classify where the index comes
        // from. A constant source makes a0 fixed for the whole draw (indexed reads then resolve
        // at evaluation time); a position/vertex source makes the transform genuinely per-vertex.
        // Diagnostic only here - the analysis still declines relative reads below.
        if (ctx.dst.id.type == DxsoRegisterType::Addr) {
          const DxsoRegister& asrc = ctx.src[0];
          addrState = CompState::Unknown;
          addrSourceReg = 0xFFFFu;
          if (!asrc.hasRelative && !inFlowControl) {
            switch (asrc.id.type) {
              case DxsoRegisterType::Const:
                addrState = CompState::Constant;
                addrSourceReg = asrc.id.num;
                break;
              case DxsoRegisterType::Temp:
                if (asrc.id.num < kMaxTempRegisters) {
                  addrState = temps[asrc.id.num].c[0];
                  for (uint32_t i = 1; i < 4; i++)
                    addrState = combine(addrState, temps[asrc.id.num].c[i]);
                }
                break;
              case DxsoRegisterType::Input:
                addrState = CompState::Affine;  // vertex-derived index
                break;
              default:
                break;
            }
          }
          continue;
        }

        // Everything from here writes a register. Work out which one, and whether the write
        // is modelable; a write that is not becomes Unknown rather than being ignored, so a
        // later read of it cannot silently produce a wrong transform.
        const bool dstIsClip =
          haveClipOutput &&
          ((usesDeclaredOutput && ctx.dst.id.type == DxsoRegisterType::Output && ctx.dst.id.num == clipOutputRegister) ||
           (!usesDeclaredOutput && ctx.dst.id.type == DxsoRegisterType::RasterizerOut && ctx.dst.id.num == RasterOutPosition));
        const bool dstIsTemp = ctx.dst.id.type == DxsoRegisterType::Temp && ctx.dst.id.num < kMaxTempRegisters;

        if (!dstIsClip && !dstIsTemp) {
          continue;  // writes something the position path can never read
        }

        RegStates& dstStates = dstIsClip ? clipOut : temps[ctx.dst.id.num];

        uint8_t dstMask = 0;
        for (uint32_t i = 0; i < 4; i++) {
          if (ctx.dst.mask[i])
            dstMask |= uint8_t(1u << i);
        }

        auto poisonDst = [&](uint8_t mask) {
          for (uint32_t i = 0; i < 4; i++) {
            if (mask & (1u << i))
              dstStates.c[i] = CompState::Unknown;
          }
        };

        const OpClass opClass = classifyOpcode(opcode);
        if (!opClass.supported || inFlowControl || ctx.instruction.predicated ||
            ctx.dst.saturate || ctx.dst.hasRelative || ctx.dst.shift > 8 || ctx.dst.shift < -8) {
          poisonDst(dstMask);
          continue;
        }

        // The matrix macros always write a fixed component range regardless of the encoded mask
        if (opClass.matrixRows != 0) {
          dstMask = uint8_t((1u << opClass.matrixRows) - 1u);
        }

        // Resolve the sources this opcode actually consumes (the decode context reuses its
        // source array across instructions, so reading past srcCount would see stale operands)
        VsClipSrc srcs[3];
        RegStates srcStates[3];
        bool sourcesResolved = true;

        for (uint32_t s = 0; s < opClass.srcCount && sourcesResolved; s++) {
          const DxsoRegister& reg = ctx.src[s];
          RegStates base;

          if (reg.hasRelative) {
            // Indexed constant read c[a0 + reg.id.num]: the exact register is a runtime value, so
            // the static analysis cannot resolve it and declines. Record why (from the first one
            // seen) so the caller can tell a recoverable indexed transform - a0 loaded from a
            // constant, fixed for the whole draw - from a genuinely per-vertex one.
            usesRelativeAddressing = true;
            if (relativeReadBaseReg == 0xFFFFu) {
              relativeReadBaseReg = reg.id.num;
              relativeAddressConstantDerived = (addrState == CompState::Constant);
              relativeAddressSourceReg = addrSourceReg;
            }
            sourcesResolved = false;
            break;
          }

          switch (reg.id.type) {
            case DxsoRegisterType::Temp: {
              if (reg.id.num >= kMaxTempRegisters) {
                sourcesResolved = false;
                break;
              }
              base = temps[reg.id.num];
              srcs[s].file = VsClipSrcFile::Temp;
              srcs[s].index = uint16_t(reg.id.num);
              break;
            }
            case DxsoRegisterType::Const: {
              // Software vertex processing exposes far more constants than the probe's
              // register window; keep the analysis inside the hardware range so the evaluator
              // can index a bounded array
              if (reg.id.num >= 4096) {
                sourcesResolved = false;
                break;
              }
              uint32_t literalIndex = UINT32_MAX;
              for (const auto& entry : literalRegisters) {
                if (entry.first == reg.id.num) {
                  literalIndex = entry.second;
                  break;
                }
              }
              if (literalIndex != UINT32_MAX) {
                srcs[s].file = VsClipSrcFile::Literal;
                srcs[s].index = uint16_t(literalIndex);
              } else {
                srcs[s].file = VsClipSrcFile::FloatConst;
                srcs[s].index = uint16_t(reg.id.num);
              }
              for (uint32_t i = 0; i < 4; i++)
                base.c[i] = CompState::Constant;
              break;
            }
            case DxsoRegisterType::Input: {
              if (!havePositionInput || reg.id.num != positionInputRegister) {
                sourcesResolved = false;
                break;
              }
              srcs[s].file = VsClipSrcFile::Position;
              srcs[s].index = 0;
              for (uint32_t i = 0; i < 4; i++)
                base.c[i] = CompState::Affine;
              break;
            }
            default:
              sourcesResolved = false;
              break;
          }

          if (!sourcesResolved)
            break;

          // The matrix macros name a constant register BASE rather than a value, so the
          // register span has to be resolvable directly
          if (opClass.matrixRows != 0 && s == 1) {
            if (srcs[s].file != VsClipSrcFile::FloatConst ||
                uint32_t(srcs[s].index) + opClass.matrixRows > 4096) {
              sourcesResolved = false;
              break;
            }
          }

          uint8_t swizzle = 0;
          for (uint32_t i = 0; i < 4; i++)
            swizzle |= uint8_t(uint32_t(reg.swizzle[i]) << (i * 2u));
          srcs[s].swizzle = swizzle;
          srcs[s].modifier = uint8_t(reg.modifier);

          for (uint32_t i = 0; i < 4; i++)
            srcStates[s].c[i] = base.c[swizzleComponent(swizzle, i)];

          if (!modifierIsAffine(reg.modifier)) {
            const bool constantOnly = modifierIsConstantOnly(reg.modifier);
            for (uint32_t i = 0; i < 4; i++) {
              srcStates[s].c[i] = (constantOnly && srcStates[s].c[i] == CompState::Constant)
                ? CompState::Constant : CompState::Unknown;
            }
          }
        }

        if (!sourcesResolved) {
          poisonDst(dstMask);
          continue;
        }

        // Which source components the opcode reads: dot products and matrix macros consume the
        // whole vector, component-wise opcodes only the ones they write
        uint8_t readMask = dstMask;
        switch (opClass.opcode) {
          case VsClipOpcode::Dp4:
          case VsClipOpcode::MatrixDp4: readMask = 0xF; break;
          case VsClipOpcode::Dp3:
          case VsClipOpcode::MatrixDp3:
          case VsClipOpcode::Nrm3:
          case VsClipOpcode::Crs:       readMask = 0x7; break;
          case VsClipOpcode::Dp2Add:    readMask = uint8_t(0x3u | dstMask); break;
          default: break;
        }

        // Result state per written component
        CompState resultStates[4] = { CompState::Unknown, CompState::Unknown,
                                      CompState::Unknown, CompState::Unknown };
        bool modelable = true;

        if (!opClass.linear) {
          // Exact only while nothing carrying position dependence reaches the operation
          for (uint32_t s = 0; s < opClass.srcCount && modelable; s++) {
            for (uint32_t i = 0; i < 4; i++) {
              if ((readMask & (1u << i)) == 0)
                continue;
              if (srcStates[s].c[i] != CompState::Constant) {
                modelable = false;
                break;
              }
            }
          }
          for (uint32_t i = 0; i < 4; i++)
            resultStates[i] = CompState::Constant;
        } else if (opClass.opcode == VsClipOpcode::Dp3 || opClass.opcode == VsClipOpcode::Dp4 ||
                   opClass.opcode == VsClipOpcode::Dp2Add) {
          const uint32_t width = opClass.opcode == VsClipOpcode::Dp4 ? 4u
                               : (opClass.opcode == VsClipOpcode::Dp3 ? 3u : 2u);
          CompState scalar = CompState::Constant;
          for (uint32_t i = 0; i < width; i++)
            scalar = combine(scalar, combineProduct(srcStates[0].c[i], srcStates[1].c[i]));
          if (opClass.opcode == VsClipOpcode::Dp2Add) {
            for (uint32_t i = 0; i < 4; i++) {
              if (dstMask & (1u << i))
                scalar = combine(scalar, srcStates[2].c[i]);
            }
          }
          modelable = scalar != CompState::Unknown;
          for (uint32_t i = 0; i < 4; i++)
            resultStates[i] = scalar;
        } else if (opClass.matrixRows != 0) {
          CompState rowState = CompState::Constant;
          const uint32_t width = opClass.opcode == VsClipOpcode::MatrixDp4 ? 4u : 3u;
          for (uint32_t i = 0; i < width; i++)
            rowState = combine(rowState, srcStates[0].c[i]);
          modelable = rowState != CompState::Unknown;
          for (uint32_t i = 0; i < 4; i++)
            resultStates[i] = rowState;
        } else {
          for (uint32_t i = 0; i < 4; i++) {
            if ((dstMask & (1u << i)) == 0)
              continue;
            CompState state = CompState::Unknown;
            switch (opClass.opcode) {
              case VsClipOpcode::Mov: state = srcStates[0].c[i]; break;
              case VsClipOpcode::Add:
              case VsClipOpcode::Sub: state = combine(srcStates[0].c[i], srcStates[1].c[i]); break;
              case VsClipOpcode::Mul: state = combineProduct(srcStates[0].c[i], srcStates[1].c[i]); break;
              case VsClipOpcode::Mad: state = combine(combineProduct(srcStates[0].c[i], srcStates[1].c[i]),
                                                      srcStates[2].c[i]); break;
              case VsClipOpcode::Lrp: {
                // dst = src2 + src0 * (src1 - src2)
                const CompState delta = combine(srcStates[1].c[i], srcStates[2].c[i]);
                state = combine(combineProduct(srcStates[0].c[i], delta), srcStates[2].c[i]);
                break;
              }
              default: state = CompState::Unknown; break;
            }
            if (state == CompState::Unknown)
              modelable = false;
            resultStates[i] = state;
          }
        }

        if (!modelable) {
          poisonDst(dstMask);
          continue;
        }

        if (ops.size() >= 4096) {
          result.status = VsClipTransformStatus::TooComplex;
          return result;
        }

        VsClipOp op;
        op.opcode = opClass.opcode;
        op.dstIsClipPosition = dstIsClip ? 1u : 0u;
        op.dstIndex = dstIsClip ? 0u : uint8_t(ctx.dst.id.num);
        op.dstMask = dstMask;
        op.srcCount = uint8_t(opClass.srcCount);
        op.dstScale = std::ldexp(1.0f, ctx.dst.shift);
        for (uint32_t s = 0; s < opClass.srcCount; s++)
          op.src[s] = srcs[s];
        ops.push_back(op);

        for (uint32_t i = 0; i < 4; i++) {
          if (dstMask & (1u << i))
            dstStates.c[i] = resultStates[i];
        }
      }
    } catch (...) {
      result.status = VsClipTransformStatus::DecodeFailed;
      return result;
    }

    result.instructionCount = instructionCount;
    result.positionInputRegister = positionInputRegister;
    result.usesRelativeAddressing = usesRelativeAddressing;
    result.relativeAddressConstantDerived = relativeAddressConstantDerived;
    result.relativeAddressSourceReg = uint16_t(relativeAddressSourceReg);
    result.relativeReadBaseReg = uint16_t(relativeReadBaseReg);

    if (preTransformed && !havePositionInput) {
      result.status = VsClipTransformStatus::PreTransformedPosition;
      return result;
    }
    if (!havePositionInput) {
      result.status = VsClipTransformStatus::NoPositionInput;
      return result;
    }
    if (!haveClipOutput) {
      result.status = VsClipTransformStatus::NoPositionOutput;
      return result;
    }

    for (uint32_t i = 0; i < 4; i++) {
      if (clipOut.c[i] == CompState::Unknown) {
        result.status = VsClipTransformStatus::PositionNotAffine;
        return result;
      }
    }

    // Backward slice: keep only the instructions clip position actually consumes. Typically a
    // handful out of a shader of a hundred, which is what makes per-draw evaluation cheap.
    std::vector<uint8_t> keep(ops.size(), 0);
    {
      uint8_t liveTemp[kMaxTempRegisters] = {};
      uint8_t liveClip = 0xF;

      for (size_t i = ops.size(); i-- > 0;) {
        const VsClipOp& op = ops[i];
        const uint8_t live = op.dstIsClipPosition ? liveClip : liveTemp[op.dstIndex];
        if ((live & op.dstMask) == 0)
          continue;

        keep[i] = 1;
        if (op.dstIsClipPosition)
          liveClip = uint8_t(liveClip & ~op.dstMask);
        else
          liveTemp[op.dstIndex] = uint8_t(liveTemp[op.dstIndex] & ~op.dstMask);

        for (uint32_t s = 0; s < op.srcCount; s++) {
          if (op.src[s].file == VsClipSrcFile::Temp)
            liveTemp[op.src[s].index] = 0xF;
        }
      }
    }

    result.ops.reserve(ops.size());
    for (size_t i = 0; i < ops.size(); i++) {
      if (keep[i])
        result.ops.push_back(ops[i]);
    }

    if (result.ops.size() > kMaxSliceOps) {
      result.ops.clear();
      result.status = VsClipTransformStatus::TooComplex;
      return result;
    }

    result.literals = std::move(literals);
    result.sliceInstructionCount = uint32_t(result.ops.size());

    // The constant registers the transform is built from - the engine-agnostic answer to
    // "where does this game keep its camera"
    for (const VsClipOp& op : result.ops) {
      const uint32_t rows = (op.opcode == VsClipOpcode::MatrixDp4 || op.opcode == VsClipOpcode::MatrixDp3)
        ? uint32_t(op.dstMask == 0x3 ? 2u : (op.dstMask == 0x7 ? 3u : 4u)) : 0u;
      for (uint32_t s = 0; s < op.srcCount; s++) {
        if (op.src[s].file != VsClipSrcFile::FloatConst)
          continue;
        const uint32_t span = (rows != 0 && s == 1) ? rows : 1;
        for (uint32_t i = 0; i < span; i++) {
          const uint32_t reg = uint32_t(op.src[s].index) + i;
          result.maxConstRegister = std::max(result.maxConstRegister, reg);
          if (result.constRegisters.size() < kMaxConstFootprint)
            result.constRegisters.push_back(uint16_t(reg));
        }
      }
    }
    std::sort(result.constRegisters.begin(), result.constRegisters.end());
    result.constRegisters.erase(std::unique(result.constRegisters.begin(), result.constRegisters.end()),
                                result.constRegisters.end());

    result.status = VsClipTransformStatus::Ok;
    return result;
  }

  namespace {

    // v = k[0]*pos.x + k[1]*pos.y + k[2]*pos.z + k[3]*pos.w + k[4]
    struct Aff {
      float k[5];
    };

    struct AffVec {
      Aff c[4];
    };

    inline Aff affConstant(float value) {
      Aff a {};
      a.k[4] = value;
      return a;
    }

    inline bool affIsConstant(const Aff& a) {
      return a.k[0] == 0.0f && a.k[1] == 0.0f && a.k[2] == 0.0f && a.k[3] == 0.0f;
    }

    inline Aff affScale(const Aff& a, float s) {
      Aff r;
      for (uint32_t i = 0; i < 5; i++)
        r.k[i] = a.k[i] * s;
      return r;
    }

    inline Aff affOffset(const Aff& a, float o) {
      Aff r = a;
      r.k[4] += o;
      return r;
    }

    inline Aff affAdd(const Aff& a, const Aff& b) {
      Aff r;
      for (uint32_t i = 0; i < 5; i++)
        r.k[i] = a.k[i] + b.k[i];
      return r;
    }

    inline Aff affSub(const Aff& a, const Aff& b) {
      Aff r;
      for (uint32_t i = 0; i < 5; i++)
        r.k[i] = a.k[i] - b.k[i];
      return r;
    }

    // At least one side carries no position dependence (guaranteed by the analysis; re-checked
    // here so a mis-analysed shader produces zero rather than nonsense)
    inline Aff affMul(const Aff& a, const Aff& b) {
      if (affIsConstant(b))
        return affScale(a, b.k[4]);
      if (affIsConstant(a))
        return affScale(b, a.k[4]);
      return affConstant(0.0f);
    }

    inline Aff applyModifier(const Aff& a, uint8_t modifier) {
      switch (static_cast<DxsoRegModifier>(modifier)) {
        case DxsoRegModifier::None:    return a;
        case DxsoRegModifier::Neg:     return affScale(a, -1.0f);
        case DxsoRegModifier::Bias:    return affOffset(a, -0.5f);
        case DxsoRegModifier::BiasNeg: return affScale(affOffset(a, -0.5f), -1.0f);
        case DxsoRegModifier::Sign:    return affOffset(affScale(a, 2.0f), -1.0f);
        case DxsoRegModifier::SignNeg: return affScale(affOffset(affScale(a, 2.0f), -1.0f), -1.0f);
        case DxsoRegModifier::Comp:    return affOffset(affScale(a, -1.0f), 1.0f);
        case DxsoRegModifier::X2:      return affScale(a, 2.0f);
        case DxsoRegModifier::X2Neg:   return affScale(a, -2.0f);
        case DxsoRegModifier::Abs:     return affConstant(std::abs(a.k[4]));
        case DxsoRegModifier::AbsNeg:  return affConstant(-std::abs(a.k[4]));
        default:                       return affConstant(0.0f);
      }
    }

  }

  bool evaluateVsClipTransform(const VsClipTransformProgram& program,
                               const Vector4* floatConstants,
                               uint32_t floatConstantCount,
                               Matrix4& outObjectToClip) {
    if (!program.ok() || floatConstants == nullptr)
      return false;
    if (program.maxConstRegister >= floatConstantCount)
      return false;

    AffVec temps[kMaxTempRegisters] = {};
    AffVec clip = {};

    AffVec position = {};
    for (uint32_t i = 0; i < 4; i++)
      position.c[i].k[i] = 1.0f;

    auto loadConstantRegister = [&](uint32_t reg) {
      AffVec v = {};
      const Vector4& value = floatConstants[reg];
      for (uint32_t i = 0; i < 4; i++)
        v.c[i] = affConstant(value[i]);
      return v;
    };

    for (const VsClipOp& op : program.ops) {
      AffVec src[3] = {};

      for (uint32_t s = 0; s < op.srcCount; s++) {
        AffVec base = {};
        switch (op.src[s].file) {
          case VsClipSrcFile::Temp:
            base = temps[op.src[s].index];
            break;
          case VsClipSrcFile::FloatConst:
            base = loadConstantRegister(op.src[s].index);
            break;
          case VsClipSrcFile::Literal: {
            if (op.src[s].index >= program.literals.size())
              return false;
            const Vector4& value = program.literals[op.src[s].index];
            for (uint32_t i = 0; i < 4; i++)
              base.c[i] = affConstant(value[i]);
            break;
          }
          case VsClipSrcFile::Position:
            base = position;
            break;
        }

        for (uint32_t i = 0; i < 4; i++)
          src[s].c[i] = applyModifier(base.c[swizzleComponent(op.src[s].swizzle, i)], op.src[s].modifier);
      }

      Aff result[4] = {};

      switch (op.opcode) {
        case VsClipOpcode::MatrixDp4:
        case VsClipOpcode::MatrixDp3: {
          const uint32_t rows = op.dstMask == 0x3 ? 2u : (op.dstMask == 0x7 ? 3u : 4u);
          const uint32_t width = op.opcode == VsClipOpcode::MatrixDp4 ? 4u : 3u;
          const uint32_t base = op.src[1].index;
          if (base + rows > floatConstantCount)
            return false;
          for (uint32_t r = 0; r < rows; r++) {
            const Vector4& row = floatConstants[base + r];
            Aff sum = affConstant(0.0f);
            for (uint32_t i = 0; i < width; i++)
              sum = affAdd(sum, affScale(src[0].c[i], row[i]));
            result[r] = sum;
          }
          break;
        }
        case VsClipOpcode::Dp3:
        case VsClipOpcode::Dp4:
        case VsClipOpcode::Dp2Add: {
          const uint32_t width = op.opcode == VsClipOpcode::Dp4 ? 4u
                               : (op.opcode == VsClipOpcode::Dp3 ? 3u : 2u);
          Aff sum = affConstant(0.0f);
          for (uint32_t i = 0; i < width; i++)
            sum = affAdd(sum, affMul(src[0].c[i], src[1].c[i]));
          for (uint32_t i = 0; i < 4; i++)
            result[i] = op.opcode == VsClipOpcode::Dp2Add ? affAdd(sum, src[2].c[i]) : sum;
          break;
        }
        default: {
          for (uint32_t i = 0; i < 4; i++) {
            if ((op.dstMask & (1u << i)) == 0)
              continue;
            const Aff& a = src[0].c[i];
            const Aff& b = src[1].c[i];
            const Aff& c = src[2].c[i];
            switch (op.opcode) {
              case VsClipOpcode::Mov: result[i] = a; break;
              case VsClipOpcode::Add: result[i] = affAdd(a, b); break;
              case VsClipOpcode::Sub: result[i] = affSub(a, b); break;
              case VsClipOpcode::Mul: result[i] = affMul(a, b); break;
              case VsClipOpcode::Mad: result[i] = affAdd(affMul(a, b), c); break;
              case VsClipOpcode::Lrp: result[i] = affAdd(c, affMul(a, affSub(b, c))); break;
              // Position-free from here: the analysis proved these only ever see constants
              case VsClipOpcode::Rcp: result[i] = affConstant(a.k[4] != 0.0f ? 1.0f / a.k[4] : 0.0f); break;
              case VsClipOpcode::Rsq: result[i] = affConstant(a.k[4] > 0.0f ? 1.0f / std::sqrt(a.k[4]) : 0.0f); break;
              case VsClipOpcode::Min: result[i] = affConstant(std::min(a.k[4], b.k[4])); break;
              case VsClipOpcode::Max: result[i] = affConstant(std::max(a.k[4], b.k[4])); break;
              case VsClipOpcode::Slt: result[i] = affConstant(a.k[4] < b.k[4] ? 1.0f : 0.0f); break;
              case VsClipOpcode::Sge: result[i] = affConstant(a.k[4] >= b.k[4] ? 1.0f : 0.0f); break;
              case VsClipOpcode::Exp: result[i] = affConstant(std::exp2(a.k[4])); break;
              case VsClipOpcode::Log: result[i] = affConstant(a.k[4] != 0.0f ? std::log2(std::abs(a.k[4])) : -FLT_MAX); break;
              case VsClipOpcode::Frc: result[i] = affConstant(a.k[4] - std::floor(a.k[4])); break;
              case VsClipOpcode::Abs: result[i] = affConstant(std::abs(a.k[4])); break;
              case VsClipOpcode::Pow: result[i] = affConstant(std::pow(std::abs(a.k[4]), b.k[4])); break;
              case VsClipOpcode::Sgn: result[i] = affConstant(a.k[4] > 0.0f ? 1.0f : (a.k[4] < 0.0f ? -1.0f : 0.0f)); break;
              case VsClipOpcode::Cmp: result[i] = affConstant(a.k[4] >= 0.0f ? b.k[4] : c.k[4]); break;
              case VsClipOpcode::Cnd: result[i] = affConstant(a.k[4] > 0.5f ? b.k[4] : c.k[4]); break;
              case VsClipOpcode::Nrm3: {
                const float lengthSquared = src[0].c[0].k[4] * src[0].c[0].k[4] +
                                            src[0].c[1].k[4] * src[0].c[1].k[4] +
                                            src[0].c[2].k[4] * src[0].c[2].k[4];
                const float scale = lengthSquared > 0.0f ? 1.0f / std::sqrt(lengthSquared) : 0.0f;
                result[i] = affConstant(a.k[4] * scale);
                break;
              }
              case VsClipOpcode::Crs: {
                const uint32_t j = (i + 1u) % 3u;
                const uint32_t k = (i + 2u) % 3u;
                result[i] = affConstant(src[0].c[j].k[4] * src[1].c[k].k[4] -
                                        src[0].c[k].k[4] * src[1].c[j].k[4]);
                break;
              }
              default: return false;
            }
          }
          break;
        }
      }

      AffVec& dst = op.dstIsClipPosition ? clip : temps[op.dstIndex];
      for (uint32_t i = 0; i < 4; i++) {
        if (op.dstMask & (1u << i))
          dst.c[i] = op.dstScale == 1.0f ? result[i] : affScale(result[i], op.dstScale);
      }
    }

    // clip_i = k_i[0]*x + k_i[1]*y + k_i[2]*z + k_i[3]*w + k_i[4], with w == 1, so the constant
    // term folds into the matrix's translation column. Matrix4 stores columns, clip = M * pos.
    for (uint32_t column = 0; column < 4; column++) {
      for (uint32_t row = 0; row < 4; row++) {
        const float value = column < 3
          ? clip.c[row].k[column]
          : clip.c[row].k[3] + clip.c[row].k[4];
        if (!std::isfinite(value))
          return false;
        outObjectToClip[column][row] = value;
      }
    }

    return true;
  }

}
