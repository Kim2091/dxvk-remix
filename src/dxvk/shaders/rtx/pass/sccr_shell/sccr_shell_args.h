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

// The per-geometry shell data SccrShellManager builds (rtx_fork_sccr_shell.cpp,
// sccr_shell_adjacency.comp.slang) and the shaders read (sccr_shell.slangh). Shared by C++ and Slang: defines and one
// argument struct only.

// ---------------------------------------------------------------------------------------------------------------------
// The shell buffer of one geometry (a BlasEntry's RaytraceGeometry), bound through the bindless raw-buffer table.
// Its slot is Surface.shellBufferIndex (data15.w). In dwords:
//   [0, 4)                header: triangle count T, vertex count V, flags, depth-limit scale (float, always 1)
//   [4, 4 + 3T)           per triangle t, edge slot e (the edge opposite vertex e, from vertex (e + 1) % 3 to
//                         (e + 2) % 3 in the triangle's winding): the neighbour across it, SCCR_SHELL_NEIGHBOUR_*
//   [4 + 3T, 4 + 3T + 2V) per vertex (indexed like the position buffer; split copies of one welded position carry the
//                         same values except SCCR_SHELL_VERTEX_FLAG_CANONICAL): word 0 the welded shell direction
//                         (object space, on the side of the triangles' cross(p1 - p0, p2 - p0) normals: out of a mesh
//                         wound that way; the inner shell lies along its negation) as 16+16-bit signed octahedral snorm;
//                         word 1 the fold guard's depth limits in object units (SCCR_SHELL_VERTEX_*LIMIT*: a half float
//                         for a shell against the direction in bits 0-15, a shortened one for a shell along it in bits
//                         20-31) and SCCR_SHELL_VERTEX_FLAG_* (bits 16-19)
//   [4 + 3T + 2V, 4 + 3T + 3V) per vertex: its welded id, the smallest vertex index at its position (the vertex's own
//                         index when it did not weld). Read by the real-geometry tessellator (sccr_tess.comp.slang), so
//                         every copy of a welded corner takes its position and height from one vertex.
// Triangle t is the t-th triangle of the geometry's whole index buffer (or of its vertices when it has none): for a hit
// it is primitiveIndex + surface.firstIndex / 3.
// ---------------------------------------------------------------------------------------------------------------------

#define SCCR_SHELL_HEADER_DWORDS 4
#define SCCR_SHELL_HEADER_TRIANGLES 0
#define SCCR_SHELL_HEADER_VERTICES 1
#define SCCR_SHELL_HEADER_FLAGS 2
#define SCCR_SHELL_HEADER_DEPTH_LIMIT_SCALE 3

// Header flags. Bits 0-7 hold the layout version, and bits 8-11 the weld mode (SCCR_SHELL_WELD_*).
#define SCCR_SHELL_VERSION 3u
#define SCCR_SHELL_FLAG_VERSION_MASK 0xFFu
#define SCCR_SHELL_FLAG_WELD_MODE_SHIFT 8u
// Positions weld when their float bits match (-0 as +0).
#define SCCR_SHELL_WELD_EXACT 0u
// Positions weld when they round to the same point of a grid whose spacing is rtx.displacement.shell.weldEpsilon
// times the mesh's bounding diagonal (SccrShellBuildArgs::weldCell).
#define SCCR_SHELL_WELD_GRID 1u
// Some edge has no neighbour (open, non-manifold, flipped winding, a degenerate direction at an end, or a corner that
// did not weld).
#define SCCR_SHELL_FLAG_HAS_BOUNDARY (1u << 16)
// Some vertex has a finite fold limit (the march clamps the shell depth there when the material's depth exceeds it).
#define SCCR_SHELL_FLAG_FOLDS_CLAMPED (1u << 17)
// A scratch hash table ran out of probes, so some corners did not weld and their edges are boundaries.
#define SCCR_SHELL_FLAG_WELD_OVERFLOW (1u << 18)
// Some welded direction summed to almost nothing (a fin, a two-sided sheet) and fell back to a face normal.
#define SCCR_SHELL_FLAG_DEGENERATE_DIRECTIONS (1u << 19)
// Some edge is shared by more than two triangles.
#define SCCR_SHELL_FLAG_NON_MANIFOLD (1u << 20)
// Some edge is shared by two triangles running along it in the same direction (mirrored or double-sided pieces).
#define SCCR_SHELL_FLAG_FLIPPED_WINDING (1u << 21)

// Neighbour words hold the neighbour triangle in bits 0-29 and the neighbour's edge slot in bits 30-31. A boundary edge
// has every triangle bit set (never a triangle, see SCCR_SHELL_MAX_TRIANGLES) and its kind in bits 30-31:
// - NONE: open, with no second triangle. This also covers an end that did not weld, or an edge collapsed to one position.
// - NON_MANIFOLD: shared by more than two triangles.
// - FLIPPED: shared by two triangles running along it the same way (mirrored or double-sided pieces).
// - DEGENERATE: an end's welded direction is degenerate (a fin or a two-sided sheet).
// The march stops at every boundary, and plain SCCR shades the hit there.
#define SCCR_SHELL_NEIGHBOUR_NONE 0xFFFFFFFFu
#define SCCR_SHELL_NEIGHBOUR_NON_MANIFOLD 0x3FFFFFFFu
#define SCCR_SHELL_NEIGHBOUR_FLIPPED 0x7FFFFFFFu
#define SCCR_SHELL_NEIGHBOUR_DEGENERATE 0xBFFFFFFFu
#define SCCR_SHELL_NEIGHBOUR_TRIANGLE_MASK 0x3FFFFFFFu
#define SCCR_SHELL_NEIGHBOUR_SLOT_SHIFT 30u
#define SCCR_SHELL_NEIGHBOUR_IS_BOUNDARY(word) \
  (((word) & SCCR_SHELL_NEIGHBOUR_TRIANGLE_MASK) == SCCR_SHELL_NEIGHBOUR_TRIANGLE_MASK)

// Vertex word 1, bits 16-19.
// The parity of the welded vertex id (the smallest vertex index at that position), for tie-breaking at crossings.
#define SCCR_SHELL_VERTEX_FLAG_WELD_PARITY (1u << 16)
// The direction is a face normal, because the welded sum was degenerate.
#define SCCR_SHELL_VERTEX_FLAG_DEGENERATE (1u << 17)
// The vertex did not weld (scratch table overflow), so it has no direction and depth limits of 0.
#define SCCR_SHELL_VERTEX_FLAG_UNWELDED (1u << 18)
// This vertex index is its position's welded id (the other copies are splits at hard edges or UV seams).
#define SCCR_SHELL_VERTEX_FLAG_CANONICAL (1u << 19)

// Vertex word 1 holds the fold guard's depth limits, in object units along the unit welded direction.
// A limit is the shell depth at which the first triangle at this position comes within SCCR_SHELL_FOLD_MARGIN of
// folding (see sccrShellFold). Bits 0-15 hold, as a half float, the limit for a shell against the welded direction
// (the march's orientation +1, the usual case). Bits 20-31 hold, as a half float's bits 3-14 rounded down, the limit for
// a shell along it (orientation -1, with the viewer on the other side of the triangles' normals).
// SCCR_SHELL_VERTEX_NO_LIMIT (the largest half float, about 65280 in the shortened form) means no triangle at the
// position inverts at any depth. 0 means the vertex did not weld.
#define SCCR_SHELL_VERTEX_NO_LIMIT 65504.0f
#define SCCR_SHELL_VERTEX_LIMIT_MASK 0xFFFFu
#define SCCR_SHELL_VERTEX_OUTWARD_LIMIT_SHIFT 20u
#define SCCR_SHELL_VERTEX_OUTWARD_LIMIT_MASK 0xFFFu
// A corner's direction may lose at most this share of its height over the inner triangle's plane, relative to depth 0.
// A triangle that shrinks evenly keeps a quarter of its area there, and one that tips over is stopped well before.
#define SCCR_SHELL_FOLD_MARGIN 0.75f

// The march's switches (RaytraceArgs::pomSccrShellFlags, from SccrShellManager::marchFlags). Bits 0, 1, 5-7 and
// 16-31 are unused.
// Each vertex's shell depth is capped at its fold limit (SCCR_SHELL_VERTEX_*LIMIT*).
#define SCCR_SHELL_MARCH_FLAG_FOLD_GUARD (1u << 2)
// Direct-light self-shadows of shell pixels march outward from the carved point through the same prisms
// (sccrShellShadowMarch). Without it they take the texture-space self-shadow march, in the carved triangle's frame
// and shell depth.
#define SCCR_SHELL_MARCH_FLAG_SHADOWS (1u << 3)
// The cones the field holds are conservative (coarse cones, or cones generated without a ray walk, see
// PomConeMapManager::conesConservative), so the outward shadow march may take cone steps. Relaxed cones only bound rays
// that enter the field from above.
#define SCCR_SHELL_MARCH_FLAG_CONSERVATIVE_CONES (1u << 4)
// Bits 8-15: rtx.displacement.shell.maxShadowPrisms, the most prisms one shadow march crosses into.
#define SCCR_SHELL_MARCH_SHADOW_PRISMS_SHIFT 8u
#define SCCR_SHELL_MARCH_SHADOW_PRISMS_MASK 0xFFu

// Geometries with more triangles get no shell data (the build packs triangle * 3 + edge slot, times 2, into 31 bits).
#define SCCR_SHELL_MAX_TRIANGLES (1u << 28)

// ---------------------------------------------------------------------------------------------------------------------
// The build (sccr_shell_adjacency.comp.slang), four dispatches per geometry over two scratch hash tables cleared to 0.
// ---------------------------------------------------------------------------------------------------------------------

// StructuredBuffer<float>: the geometry's vertex buffer (positions at positionOffset + index * positionStride bytes).
#define SCCR_SHELL_BINDING_POSITION_RO 0
// StructuredBuffer<uint>: the geometry's index buffer (unused without indices).
#define SCCR_SHELL_BINDING_INDEX_RO 1
// RWStructuredBuffer<uint>: welded positions, SCCR_SHELL_VERTEX_SLOT_DWORDS per slot.
#define SCCR_SHELL_BINDING_VERTEX_TABLE 2
// RWStructuredBuffer<uint>: welded edges, SCCR_SHELL_EDGE_SLOT_DWORDS per slot.
#define SCCR_SHELL_BINDING_EDGE_TABLE 3
// RWStructuredBuffer<uint>: the shell buffer being built (header written by the host before the dispatches).
#define SCCR_SHELL_BINDING_OUTPUT 4

// Phase 1, a thread per triangle, welds its corners by position and accumulates angle-weighted face normals and the
// shortest edge at each welded position.
#define SCCR_SHELL_PHASE_WELD 1u
// Phase 2, a thread per triangle edge, inserts the edge (its two welded ids) into the edge table.
#define SCCR_SHELL_PHASE_EDGES 2u
// Phase 3, a thread per triangle edge, writes the neighbour word.
#define SCCR_SHELL_PHASE_RESOLVE 3u
// Phase 4, a thread per vertex, writes the vertex words.
#define SCCR_SHELL_PHASE_VERTICES 4u
// Phase 5, dispatched right after phase 1 with a thread per triangle, folds the depths at which the triangle's prism
// inverts, for both orientations, into its corners' welded positions (the fold guard).
#define SCCR_SHELL_PHASE_FOLD 5u

#define SCCR_SHELL_GROUP_SIZE 128
// Dispatches are 2D: rows of at most this many groups, so very large meshes stay within the group count limit.
#define SCCR_SHELL_MAX_GROUPS_PER_ROW 32768

// Vertex table slot: [0] 1 + the welded position's x bits (0: empty), [1] ~(smallest vertex index) (InterlockedMax),
// [2..4] the fixed-point direction sum (int bits), [5] ~(bits of the fold limit against the direction) (InterlockedMax,
// 0: none), [6] 1 + the triangle that claimed the slot, [7] ~(bits of the fold limit along the direction) (as [5]),
// [8] and [9] 1 + its y and z bits (claimed after [0], in that order).
#define SCCR_SHELL_VERTEX_SLOT_DWORDS 10u
// Edge table slot: [0] 1 + smaller welded id, [1] 1 + larger welded id, [2] and [3] the two sides
// (1 + ((triangle * 3 + edge slot) * 2 + direction bit)), [4] bit 0: a third side arrived (non-manifold).
#define SCCR_SHELL_EDGE_SLOT_DWORDS 5u

struct SccrShellBuildArgs {
  // Bytes from the start of the bound buffers (slice offset + offset from slice).
  uint positionOffset;
  uint positionStride;
  uint indexOffset;
  // 0: no index buffer (triangle t is vertices 3t..3t+2), 2: uint16 indices, 4: uint32 indices.
  uint indexStride;
  uint numTriangles;
  uint numVertices;
  // Slot counts, powers of two.
  uint vertexTableSize;
  uint edgeTableSize;
  uint phase;
  // Threads in one row of the 2D dispatch (groups per row times SCCR_SHELL_GROUP_SIZE).
  uint rowThreads;
  // The thread count of this phase (triangles, edges or vertices).
  uint threadCount;
  // The SCCR_SHELL_WELD_GRID spacing positions round to before they are hashed, in object units. 0 means exact bits.
  float weldCell;
};
