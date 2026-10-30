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

// The cone map passes of PomConeMapManager (rtx_fork_pom_cone_map.cpp).

// Texture2D: the source height texture (all of its mips), a pyramid level, or the whole pyramid.
#define POM_CONE_BINDING_SOURCE 0
// RWTexture2D: the level being written.
#define POM_CONE_BINDING_OUTPUT 1
// Texture2D: generated cones (R), read by POM_CONE_COPY_HEIGHT_CONE, coarse cones per cell, read by
// POM_CONE_COPY_HEIGHT_COARSE, or the cells of pom_cone_coarse (POM_CONE_COARSE_CONES).
#define POM_CONE_BINDING_CONE 2
// Texture2D: the map's level (POM_CONE_LEVEL), R = the top of the height field, read by the copy modes when level is 1.
#define POM_CONE_BINDING_LEVEL 3

// pom_cone_copy modes.
// output = (source.r at sourceMip, 0)
#define POM_CONE_COPY_HEIGHT 0
// output = (source.r at sourceMip, cone.r)
#define POM_CONE_COPY_HEIGHT_CONE 1
// output.r = the largest of the 2x2 texels under it in mip 0 of source (a one-mip view of sourceExtent).
#define POM_CONE_REDUCE_MAX 2
// output = (source.r at sourceMip, cone.r of the cell holding the texel, cells of 2^coarseShift texels)
#define POM_CONE_COPY_HEIGHT_COARSE 3
// output (1x1) = (the top of source's height field, 0): the brightest texel of its first mip of at most
// POM_CONE_LEVEL_GRID x POM_CONE_LEVEL_GRID texels, or of a grid of cell means of its last mip if it has none. One group.
#define POM_CONE_LEVEL 4
#define POM_CONE_LEVEL_GRID 16

struct PomConeCopyArgs {
  // The whole level written. The dispatch covers the texels from origin on, and a merge of single tiles writes only those.
  uvec2 extent;
  uvec2 sourceExtent;
  uint sourceMip;
  uint mode;
  uvec2 origin;
  uint coarseShift;
  // 1: the copy modes write height + 1 - top (at most 1), with top from POM_CONE_BINDING_LEVEL, so the top of the field
  // is the surface. 0: the height as it is.
  uint level;
};

// pom_cone_generate: the cone ratios of one tile of mip 0.
#define POM_CONE_TILE_SIZE 64
#define POM_CONE_GENERATE_GROUP_SIZE 8
// Pyramid reads per texel after which the hierarchical search stops refining cells and bounds them as they are, which
// only narrows the cone.
#define POM_CONE_MAX_VISITS 2048

struct PomConeGenerateArgs {
  uvec2 extent;
  uvec2 tileOrigin;
  // Candidates within nearRadius texels get the relaxed constraint, walking their ray up to rayStepsPerCandidate texels.
  // 0 steps gives conservative cones (PomConeMapManager::ConeVariant::Conservative), and with a pyramid the near field is
  // then left to the hierarchical search.
  uint nearRadius;
  uint rayStepsPerCandidate;
  // Max-height pyramid levels above mip 0. It is log2(width) on square power-of-two maps, where the pyramid reaches 1x1
  // and a hierarchical search finds every texel outside the near field. It is 0 otherwise, and the window bounds them.
  uint farLevels;
  // 255 or 65535: the cone texture's quantisation, rounded down so the stored cone never exceeds the computed one.
  float coneQuantization;
};

// pom_cone_coarse: conservative cones per cell of 2^coarseShift texels, for every texel of a map until its tiles are
// generated (square power-of-two maps only).
// One group per cell writes Output = (largest depth difference between 8-neighbours with the first one in the cell,
// largest depth in the cell). It reads mip 0 of the max-height pyramid.
#define POM_CONE_COARSE_CELLS 0
// One thread per cell writes Output.r = the cell's conservative cone, quantised like the generator's. It reads the cells
// (POM_CONE_BINDING_CONE) and the max-height pyramid from level coarseShift to topLevel.
#define POM_CONE_COARSE_CONES 1
#define POM_CONE_COARSE_GROUP_SIZE 8
// Cells are at least 8x8 texels. Maps of more than 2048 texels get larger cells, so a map never has more than 256x256.
#define POM_CONE_COARSE_MIN_SHIFT 3
#define POM_CONE_COARSE_MAX_CELLS_LOG2 8
// The far field tests the rings of cells 2..POM_CONE_COARSE_RING around the cell at every pyramid level.
#define POM_CONE_COARSE_RING 3

struct PomConeCoarseArgs {
  uvec2 extent;
  uint coarseShift;
  // The coarsest pyramid level read: the first with POM_CONE_COARSE_RING cells reaching a texture width.
  uint topLevel;
  uint mode;
  // As PomConeGenerateArgs::coneQuantization.
  float coneQuantization;
  uint pad0;
  uint pad1;
};
