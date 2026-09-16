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
  // A path that arrived by a specular lobe only reaches the cache because allowSpecularPaths
  // waived the lobe check. Serving one from an isotropic cache on a smooth surface glows, so
  // those paths get their own, stricter roughness floor.
  float minRoughnessSpecular;
  // Cells are readable once accumulatedSampleNum exceeds this. The SDK default of 0 lets a
  // cell answer from a single sample, which is what makes newly revealed geometry glow: one
  // bright path lands in an empty cell and is read back as if it had converged.
  uint minSampleCount;
  uint pad2;
};
#ifdef __cplusplus
static_assert(sizeof(SharcArgs) == 80);
#endif
