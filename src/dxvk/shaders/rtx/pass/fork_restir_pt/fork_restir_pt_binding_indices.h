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

// NEE cache. The global NAMES are fixed by nee_cache.h, which references them
// unconditionally from its struct methods -- the same five
// integrate_indirect_bindings.slangh:170-182 declares. Phase 2 pulls the header
// in for two things only: the emissive-hit MIS weight and the emissive-hit task
// feedback. NEE-cache *sampling* inside the kernel stays out of scope.
#define FORK_RESTIR_PT_BINDING_NEE_CACHE                                115
#define FORK_RESTIR_PT_BINDING_NEE_CACHE_SAMPLE                         116
#define FORK_RESTIR_PT_BINDING_NEE_CACHE_TASK                           117
#define FORK_RESTIR_PT_BINDING_NEE_CACHE_THREAD_TASK                    118
#define FORK_RESTIR_PT_BINDING_PRIMITIVE_ID_PREFIX_SUM                  119

// Inputs / Outputs
// Debug-only parity buffer: one float4 per padded pixel holding the traced
// path's radiance in .xyz and its packed terminal descriptor in .w. The trace
// dispatch writes it; the replay-verify dispatch reads it back and compares.
// Allocated only while rtx.restirPT.enableDebugTrace is on.
#define FORK_RESTIR_PT_BINDING_PARITY_INPUT_OUTPUT                      120

// The integrate_direct -> integrate_indirect handoff, read verbatim by the
// PSR / secondary-selected continuation entry point
// (integrator_indirect.slangh:147-160). SecondaryConeRadius is the miss test for
// those pixels (integrate_indirect.slangh:118).
#define FORK_RESTIR_PT_BINDING_RAY_ORIGIN_DIRECTION_INPUT               121
#define FORK_RESTIR_PT_BINDING_THROUGHPUT_CONE_RADIUS_INPUT             122
#define FORK_RESTIR_PT_BINDING_FIRST_SAMPLED_LOBE_DATA_INPUT            123
#define FORK_RESTIR_PT_BINDING_SHARED_MEDIUM_MATERIAL_INDEX_INPUT       124
#define FORK_RESTIR_PT_BINDING_SECONDARY_CONE_RADIUS_INPUT              125

// The two real-mode outputs. IndirectRadianceHitDistance is the slot
// integrate_indirect would have written; integrate_nee reads it immediately
// after (integrate_nee.comp.slang:110-131) and demodulate folds it for
// secondary-selected pixels (demodulate.comp.slang:776-798).
#define FORK_RESTIR_PT_BINDING_INDIRECT_RADIANCE_HIT_DISTANCE_OUTPUT    126
#define FORK_RESTIR_PT_BINDING_RESERVOIR_OUTPUT                         127

// --- Final shading pass (separate pipeline, own descriptor set) -------------
// Deliberately does NOT define RAB_HAS_CURRENT_GBUFFER: shading a phase-2
// reservoir is `F * weight` routed by a stored flag bit, so no surface
// interaction is needed. The primary vertex's BSDF is already folded into F by
// the trace kernel's own primary scatter.
#define FORK_RESTIR_PT_FS_BINDING_SHARED_FLAGS_INPUT                    130
#define FORK_RESTIR_PT_FS_BINDING_PRIMARY_CONE_RADIUS_INPUT             131
#define FORK_RESTIR_PT_FS_BINDING_RESERVOIR_INPUT                       132
#define FORK_RESTIR_PT_FS_BINDING_PRIMARY_INDIRECT_DIFFUSE_INPUT_OUTPUT  133
#define FORK_RESTIR_PT_FS_BINDING_PRIMARY_INDIRECT_SPECULAR_INPUT_OUTPUT 134

// --- Spatial reuse pass (phase 3; separate pipeline, own descriptor set) -----
// The same 13-entry primary G-buffer set the trace pass binds at 100-112, in new
// slots: RAB_GetGBufferSurface reads those globals BY NAME under
// RAB_HAS_CURRENT_GBUFFER, so the names are fixed and only the numbers differ.
// Deliberately absent: the NEE cache (the reconnection shift never evaluates it --
// see the accepted-bias note in the spatial pass), the sky probe (the shift reads
// stored irradiance, it never re-evaluates the sky) and the parity buffer.
//
// 135-152 was re-grepped against every src/dxvk/shaders/rtx/pass/**/
// *_binding_indices.h before being claimed; bindings are per-pipeline, and within
// this pipeline the only other occupants are COMMON_RAYTRACING_BINDINGS (0-19 and
// the fork atmosphere range 200-215).
#define FORK_RESTIR_PT_SR_BINDING_WORLD_SHADING_NORMAL_INPUT               135
#define FORK_RESTIR_PT_SR_BINDING_PERCEPTUAL_ROUGHNESS_INPUT               136
#define FORK_RESTIR_PT_SR_BINDING_HIT_DISTANCE_INPUT                       137
#define FORK_RESTIR_PT_SR_BINDING_ALBEDO_INPUT                             138
#define FORK_RESTIR_PT_SR_BINDING_BASE_REFLECTIVITY_INPUT                  139
#define FORK_RESTIR_PT_SR_BINDING_WORLD_POSITION_INPUT                     140
#define FORK_RESTIR_PT_SR_BINDING_VIEW_DIRECTION_INPUT                     141
#define FORK_RESTIR_PT_SR_BINDING_CONE_RADIUS_INPUT                        142
#define FORK_RESTIR_PT_SR_BINDING_POSITION_ERROR_INPUT                     143
#define FORK_RESTIR_PT_SR_BINDING_SHARED_FLAGS_INPUT                       144
#define FORK_RESTIR_PT_SR_BINDING_SHARED_SURFACE_INDEX_INPUT               145
#define FORK_RESTIR_PT_SR_BINDING_SUBSURFACE_DATA_INPUT                    146
#define FORK_RESTIR_PT_SR_BINDING_SUBSURFACE_DIFFUSION_PROFILE_DATA_INPUT  147

// Reservoir ping-pong. Both are slices of the SAME two-page buffer; the host
// swaps which page each one points at per round, so the shader keeps addressing
// through restirPtReservoirIndex and never learns about paging.
#define FORK_RESTIR_PT_SR_BINDING_RESERVOIR_INPUT                          148
#define FORK_RESTIR_PT_SR_BINDING_RESERVOIR_OUTPUT                         149

// --- Temporal reuse pass (phase 4; separate pipeline, own descriptor set) ----
// The same 13-entry primary G-buffer set again, for the same by-name reason, plus
// three history / reprojection inputs the spatial pass does not need.
//
// Note what is NOT here: the surface mapping buffer that remaps a history
// reservoir's reconnection-vertex surface index into this frame's surface list.
// It needs no slot -- it is already bound at BINDING_SURFACE_MAPPING_BUFFER (5) by
// COMMON_RAYTRACING_BINDINGS, which this pipeline binds.
#define FORK_RESTIR_PT_TR_BINDING_WORLD_SHADING_NORMAL_INPUT               150
#define FORK_RESTIR_PT_TR_BINDING_PERCEPTUAL_ROUGHNESS_INPUT               151
#define FORK_RESTIR_PT_TR_BINDING_HIT_DISTANCE_INPUT                       152
#define FORK_RESTIR_PT_TR_BINDING_ALBEDO_INPUT                             153
#define FORK_RESTIR_PT_TR_BINDING_BASE_REFLECTIVITY_INPUT                  154
#define FORK_RESTIR_PT_TR_BINDING_WORLD_POSITION_INPUT                     155
#define FORK_RESTIR_PT_TR_BINDING_VIEW_DIRECTION_INPUT                     156
#define FORK_RESTIR_PT_TR_BINDING_CONE_RADIUS_INPUT                        157
#define FORK_RESTIR_PT_TR_BINDING_POSITION_ERROR_INPUT                     158
#define FORK_RESTIR_PT_TR_BINDING_SHARED_FLAGS_INPUT                       159
#define FORK_RESTIR_PT_TR_BINDING_SHARED_SURFACE_INDEX_INPUT               160
#define FORK_RESTIR_PT_TR_BINDING_SUBSURFACE_DATA_INPUT                    161
#define FORK_RESTIR_PT_TR_BINDING_SUBSURFACE_DIFFUSION_PROFILE_DATA_INPUT  162

// History / reprojection. The previous world position texture is the ONLY exact
// previous-frame surface datum this port relies on -- see THE TEMPORAL SOURCE
// SURFACE block in fork_restir_pt_temporal_reuse.comp.slang for why GBufferLast is
// deliberately not used despite carrying a previous shading normal.
#define FORK_RESTIR_PT_TR_BINDING_PREV_WORLD_POSITION_INPUT                163
#define FORK_RESTIR_PT_TR_BINDING_MVEC_INPUT                               164
#define FORK_RESTIR_PT_TR_BINDING_GRADIENTS_INPUT                          165

// Reservoirs. The current page is read AND written in place (each lane touches
// only its own index, so there is no race); the history page is last frame's final
// page and is read at the reprojected pixel. They are always different pages of
// the three-page allocation -- see the rotation comment on m_historyPage.
#define FORK_RESTIR_PT_TR_BINDING_RESERVOIR_HISTORY_INPUT                  166
#define FORK_RESTIR_PT_TR_BINDING_RESERVOIR_INPUT_OUTPUT                   167

#define FORK_RESTIR_PT_MIN_BINDING   FORK_RESTIR_PT_BINDING_WORLD_SHADING_NORMAL_INPUT
#define FORK_RESTIR_PT_MAX_BINDING   FORK_RESTIR_PT_TR_BINDING_RESERVOIR_INPUT_OUTPUT

// Size of one RestirPtReservoir element, in bytes. Shared with the host so the
// buffer allocation and the shader's structured-buffer stride can never drift.
//
// 96, not the reference's advertised 88: a Vulkan structured buffer aligns every
// vec3 to 16 B and rounds the struct size up to a multiple of 16. The layout is
// written so that padding is EXPLICIT (every vec3 is followed by a scalar) --
// see the MEMORY LAYOUT note on RestirPtReservoir.
#define RESTIR_PT_RESERVOIR_SIZE_BYTES 96

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
//
// DUAL USE (phase 3): the spatial reuse pass dispatches the same struct with
// `mode` carrying its ROUND INDEX instead, for the same reason -- 1..N rounds are
// N dispatches inside one frame. The two passes are different pipelines and never
// read each other's push constants, so the field is reused rather than widened;
// widening it would mean growing a push-constant block that is otherwise exactly
// the 16 bytes every backend guarantees.
#define FORK_RESTIR_PT_MODE_TRACE         0u
#define FORK_RESTIR_PT_MODE_REPLAY_VERIFY 1u

struct ForkReSTIRPTArgs {
  uint mode;   // trace pass: FORK_RESTIR_PT_MODE_*.  spatial pass: round index.
  // Debug-view-only. Spatial pass: the self-shift parity threshold for debug view
  // 887 (see restirPtShiftParityThreshold). Rides here rather than in RaytraceArgs
  // because that struct may only grow in complete 4-scalar groups and this is one
  // debug scalar; the trace pass zero-fills the struct, so it reads 0 there.
  float debugParam;
  uint pad1;
  uint pad2;
};
