// Copyright (c) 2026, NVIDIA CORPORATION. SPDX-License-Identifier: MIT
#pragma once
#include "rtx/utility/shader_types.h"

struct SharcArgs {
  vec3 cameraPosition;
  uint capacity;
  vec3 cameraPositionPrev;
  float gridScale;
  uint accumulationFrames;
  uint staleFrames;
  uint updateTileSize;
  uint updateBounces;
  float radianceScale;
  float minRoughness;
  uint enabled;
  uint allowSpecularPaths;
  // Emissive surfaces are excluded because the cache stores reflected radiance and the
  // path adds emission separately. A strict "any emission at all" test disqualifies every
  // surface carrying a faint emissive map, which in Portal RTX is most of them, so compare
  // luminance against a threshold instead. Zero reproduces the original behaviour.
  float maxEmissiveLuminance;
  uint pad0;
  uint pad1;
  uint pad2;
};
#ifdef __cplusplus
static_assert(sizeof(SharcArgs) == 80);
#endif
