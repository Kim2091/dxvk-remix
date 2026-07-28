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

#include "rtx/pass/common_binding_indices.h"
#include "rtx/utility/shader_types.h"

// ReSTIR PT (fork) -- pass-local binding band.
//
// Slot choice: a fresh 100-120 band, deliberately disjoint from every other set
// this pipeline binds. The pipeline binds COMMON_RAYTRACING_BINDINGS, which
// covers 0-19 (base common) plus the fork atmosphere range
// BINDING_ATMOSPHERE_MIN..BINDING_ATMOSPHERE_MAX (200-215, with 203 and 216
// retired -- common_binding_indices.h:52-136). 100-120 sits clear of both, and
// clear of ReSTIR GI's 22-40 / 61-62 band so the two are never confusable in a
// capture. The static assert below is the same guard restir_gi_reuse_binding_indices.h:52-56
// uses; the atmosphere assert is the one sparse rendering added for the same
// reason.

// Inputs -- the primary G-buffer set RAB_GetGBufferSurface reads
// (RtxdiApplicationBridge.slangh:286-324). Cloned from the ReSTIR GI reuse pass
// list minus the pieces only reuse needs (motion vectors, gradients, GI radiance
// and hit geometry, previous world position).
#define FORK_RESTIR_PT_BINDING_WORLD_SHADING_NORMAL_INPUT               100
#define FORK_RESTIR_PT_BINDING_PERCEPTUAL_ROUGHNESS_INPUT               101
#define FORK_RESTIR_PT_BINDING_HIT_DISTANCE_INPUT                       102
#define FORK_RESTIR_PT_BINDING_ALBEDO_INPUT                             103
#define FORK_RESTIR_PT_BINDING_BASE_REFLECTIVITY_INPUT                  104
#define FORK_RESTIR_PT_BINDING_WORLD_POSITION_INPUT                     105
#define FORK_RESTIR_PT_BINDING_VIEW_DIRECTION_INPUT                     106
#define FORK_RESTIR_PT_BINDING_CONE_RADIUS_INPUT                        107
#define FORK_RESTIR_PT_BINDING_POSITION_ERROR_INPUT                     108
#define FORK_RESTIR_PT_BINDING_SHARED_FLAGS_INPUT                       109
#define FORK_RESTIR_PT_BINDING_SHARED_SURFACE_INDEX_INPUT               110
#define FORK_RESTIR_PT_BINDING_SUBSURFACE_DATA_INPUT                    111
#define FORK_RESTIR_PT_BINDING_SUBSURFACE_DIFFUSION_PROFILE_DATA_INPUT  112

// Sky sources for the miss path, mirroring the indirect integrator's three-way
// branch (integrator_indirect.slangh:376-401).
#define FORK_RESTIR_PT_BINDING_LINEAR_WRAP_SAMPLER                      113
#define FORK_RESTIR_PT_BINDING_SKYPROBE                                 114

// Inputs / Outputs
// Phase-1-only parity buffer: one float4 per padded pixel holding the traced
// path's radiance in .xyz and its packed terminal descriptor in .w. The trace
// dispatch writes it; the replay-verify dispatch reads it back and compares.
// Retired once reservoirs land in phase 2 (the reservoir carries the identity).
#define FORK_RESTIR_PT_BINDING_PARITY_INPUT_OUTPUT                      120

#define FORK_RESTIR_PT_MIN_BINDING   FORK_RESTIR_PT_BINDING_WORLD_SHADING_NORMAL_INPUT
#define FORK_RESTIR_PT_MAX_BINDING   FORK_RESTIR_PT_BINDING_PARITY_INPUT_OUTPUT

#if FORK_RESTIR_PT_MIN_BINDING <= COMMON_MAX_BINDING
#error "Increase the base index of ReSTIR PT bindings to avoid overlap with common bindings!"
#endif

#if FORK_RESTIR_PT_MAX_BINDING >= BINDING_ATMOSPHERE_MIN
#error "ReSTIR PT bindings overlap the fork atmosphere binding range!"
#endif

// Push constants.
//
// The trace and replay-verify modes are two dispatches of the same shader
// *within one frame*, and RaytraceArgs is uploaded once per frame, so the mode
// selector cannot live in the constant buffer -- it has to be per-dispatch
// state. Everything frame-constant (bounce cap, roughness thresholds, the
// Russian roulette bit) stays in RaytraceArgs::restirPt*; only the mode rides
// here.
#define FORK_RESTIR_PT_MODE_TRACE         0u
#define FORK_RESTIR_PT_MODE_REPLAY_VERIFY 1u

struct ForkReSTIRPTArgs {
  uint mode;   // FORK_RESTIR_PT_MODE_*
  uint pad0;
  uint pad1;
  uint pad2;
};
