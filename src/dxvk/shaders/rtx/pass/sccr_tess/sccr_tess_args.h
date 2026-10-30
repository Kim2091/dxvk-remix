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

#include "rtx/utility/shader_types.h"

// Arguments of the SCCR real geometry tessellator, which SccrTessManager runs once per displaced static mesh
// (rtx_fork_sccr_tess.cpp, sccr_tess.comp.slang). C++ and Slang share this file,
// so it holds only defines and one argument struct.
//
// Every base triangle is cut into a regular grid of N segments per edge. N is the same for the whole mesh, so two
// triangles always agree on a shared edge. Each triangle has N^2 micro-triangles over (N + 1)(N + 2) / 2 lattice
// points. Lattice point (a, b), with a + b <= N, sits at barycentrics (b0, b1, b2) = ((N - a - b) / N, a / N, b / N).
// Each triangle writes its own copy of its lattice points, with no sharing across triangles. The copies of a point on
// a shared edge or corner are computed from the same inputs in the same order, so they are bit-identical and the
// surface is watertight.
//
// A point is displaced like the shell's prism at depth h (sccr_shell_march.slangh): X = P(b) - h(uv) * S(b).
// - P is the base surface, either linear or a Phong patch of the welded directions.
// - S(b) = sum b_i s_i interpolates the shell vectors. Each is the welded direction towards the mesh's outside, times
//   the world depth in object units, capped by the fold guard.
// - h is 1 minus the height texture's red channel, so 0 is the mesh surface and 1 the inner shell.

// StructuredBuffer<float>: the geometry's position buffer.
#define SCCR_TESS_BINDING_POSITION_RO 0
// StructuredBuffer<float>: the geometry's normal buffer (float3, or packed as SCCR_TESS_NORMAL_ENCODED).
#define SCCR_TESS_BINDING_NORMAL_RO 1
// StructuredBuffer<float>: the geometry's texture coordinates (float2).
#define SCCR_TESS_BINDING_TEXCOORD_RO 2
// StructuredBuffer<uint>: the geometry's vertex colors (one packed 8-bit RGBA word each), when it has them.
#define SCCR_TESS_BINDING_COLOR_RO 3
// StructuredBuffer<uint>: the geometry's index buffer (unused without indices).
#define SCCR_TESS_BINDING_INDEX_RO 4
// StructuredBuffer<uint>: the mesh's shell data (sccr_shell_args.h, layout SCCR_SHELL_VERSION).
#define SCCR_TESS_BINDING_SHELL_RO 5
// Sampler2D<float4>: the material's height texture (red: height, 1 = the mesh surface).
#define SCCR_TESS_BINDING_HEIGHT 6
// RWStructuredBuffer<uint>: the output vertices, SCCR_TESS_VERTEX_DWORDS each.
#define SCCR_TESS_BINDING_VERTEX_OUTPUT 7
// RWStructuredBuffer<uint>: the output index buffer, 32-bit.
#define SCCR_TESS_BINDING_INDEX_OUTPUT 8

// The output vertex holds the position (3 floats), the unit object-space normal (3 floats), the texture coordinates
// (2 floats, as the geometry's, before the texture transform) and the color (1 packed word, 0xFFFFFFFF without colors).
#define SCCR_TESS_VERTEX_DWORDS 9u
#define SCCR_TESS_VERTEX_POSITION_OFFSET 0u
#define SCCR_TESS_VERTEX_NORMAL_OFFSET 12u
#define SCCR_TESS_VERTEX_TEXCOORD_OFFSET 24u
#define SCCR_TESS_VERTEX_COLOR_OFFSET 32u

// Phase 1 runs a thread per output vertex and displaces it.
#define SCCR_TESS_PHASE_VERTICES 1u
// Phase 2 runs a thread per output triangle and writes its three indices.
#define SCCR_TESS_PHASE_INDICES 2u

// The most segments per base edge. rtx.displacement.realGeometry.maxSegments is clamped to it.
#define SCCR_TESS_MAX_SEGMENTS 64u

// SccrTessArgs::flags.
// The normal buffer holds one R32_UINT word per vertex, an unsigned octahedral direction as two 16-bit unorms.
#define SCCR_TESS_NORMAL_ENCODED (1u << 0)
// The geometry has an index buffer of 16-bit indices. Without this flag or SCCR_TESS_NO_INDICES, indices are 32-bit.
#define SCCR_TESS_INDICES_16 (1u << 1)
// The geometry has no index buffer, so triangle t is vertices 3t..3t+2.
#define SCCR_TESS_NO_INDICES (1u << 2)
// The geometry has vertex colors.
#define SCCR_TESS_HAS_COLOR (1u << 3)
// Caps each corner's shell depth at its fold limit (rtx.displacement.shell.foldGuard).
#define SCCR_TESS_FOLD_GUARD (1u << 4)

#define SCCR_TESS_GROUP_SIZE 128
// Dispatches are 2D, with rows of at most this many groups, so very large meshes stay within the group count limit.
#define SCCR_TESS_MAX_GROUPS_PER_ROW 32768

struct SccrTessArgs {
  // Offsets are bytes from the start of the bound buffers (slice offset + offset from slice). Strides are in bytes.
  uint positionOffset;
  uint positionStride;
  uint normalOffset;
  uint normalStride;

  uint texcoordOffset;
  uint texcoordStride;
  uint colorOffset;
  uint colorStride;

  uint indexOffset;
  uint numTriangles;
  uint numVertices;
  // N, the number of segments per base edge.
  uint segments;

  uint phase;
  // Threads in one row of the 2D dispatch (groups per row times SCCR_TESS_GROUP_SIZE).
  uint rowThreads;
  // The thread count of this phase (output vertices or output triangles).
  uint threadCount;
  uint flags;

  // The world length of one object unit along each object axis, from the instance transform's column lengths.
  // The mesh is built for one scale, which the cache key quantizes. The shell vector of a corner is its world depth
  // over |scale * direction|, like the march's D / |M n|.
  vec3 objectScale;
  // D, the material's world depth: (|displaceIn * inFactor| + displaceOut * outFactor) x referenceWorldUnitsPerUv.
  float worldDepth;

  // +1 when the welded directions point out of the mesh, so the inner shell lies against them.
  // -1 when they point into it.
  float orientation;
  // The height texture's mip level. It is the same for the whole mesh, so both sides of every edge sample alike.
  float heightLod;
  // The Phong tessellation shape factor. 0 keeps flat triangles, and 0.75 is the value commonly used.
  float phongShape;
  uint pad0;
};
