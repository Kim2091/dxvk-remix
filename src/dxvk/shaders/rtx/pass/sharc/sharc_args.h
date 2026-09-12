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
  uint debugMode;
};
#ifdef __cplusplus
static_assert(sizeof(SharcArgs) == 64);
#endif
