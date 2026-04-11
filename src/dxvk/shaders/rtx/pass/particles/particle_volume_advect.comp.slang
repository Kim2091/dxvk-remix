/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

// Semi-Lagrangian advection compute shader for particle volumes.
// Dispatched twice per frame:
//   Pass A (advectVelocity == 0): advects packed density4 + combustion + NvFlow damping.
//   Pass B (advectVelocity == 1): advects velocity + NvFlow damping + buoyancy + wind + vorticity.
// One thread per volume cell.
//
// The combustion model, damping formula, buoyancy force model, and
// blend-toward-target emitter coupling used below are derived from NVIDIA Flow
// (https://github.com/NVIDIA-Omniverse/PhysX/tree/main/flow), which is licensed
// under BSD-3-Clause. See ThirdPartyLicenses.txt (section "NVIDIA Flow") for
// the full license text and attribution. Specifically, this file is inspired
// by the AdvectionDensity2CS and AdvectionCombustionVelocity passes.

#include "rtx/pass/particles/particle_volume_common.h"
#include "rtx/pass/particles/particle_volume_bindings.slangh"
#include "rtx/pass/particles/particle_volume_helpers.slangh"

// Read-only inputs (previous-frame snapshots for semi-Lagrangian backtrace)
[[vk::binding(PARTICLE_VOLUME_BINDING_DENSITY4_INPUT)]]
Texture3D<vec4> Density4Input;

[[vk::binding(PARTICLE_VOLUME_BINDING_VELOCITY_INPUT)]]
Texture3D<vec4> VelocityInput;

[[vk::binding(PARTICLE_VOLUME_BINDING_OBSTACLE_INPUT)]]
Texture3D<float> ObstacleInput;

[[vk::binding(PARTICLE_VOLUME_BINDING_LINEAR_SAMPLER)]]
SamplerState LinearSampler;

// Read-write outputs (current-frame results)
[[vk::binding(PARTICLE_VOLUME_BINDING_DENSITY4_OUTPUT)]]
RWTexture3D<vec4> Density4Output;

[[vk::binding(PARTICLE_VOLUME_BINDING_VELOCITY_OUTPUT)]]
RWTexture3D<vec4> VelocityOutput;

struct AdvectPushConstants
{
  uint advectVelocity; // 0 = advect density4, 1 = advect velocity
  uint useMacCormack;  // 0 = basic semi-Lagrangian, 1 = MacCormack correction
};

[[vk::push_constant]] AdvectPushConstants pushConst;

// Compute a boundary fade factor: 1.0 in the interior, falling smoothly to
// 0.0 at the grid edges.  This implements open-boundary behavior similar to
// NvFlow's implicit dissipation when local fetch returns zero outside
// allocated sparse blocks.  Without this, clamping the backtrace to the AABB
// causes edge cells to recirculate their own values indefinitely.
float particleVolumeBoundaryFade(uvec3 cell, ParticleVolumeConstants constants)
{
  // Fade zone width in cells (about 10% of the grid, minimum 2 cells)
  const float fadeWidth = max(float(constants.gridDimension.x) * 0.1f, 2.0f);
  float fade = 1.0f;

  for (int axis = 0; axis < 3; axis++)
  {
    const float coord = float(cell[axis]);
    const float dim   = float(constants.gridDimension[axis]);
    // Distance from the nearest edge in cells
    const float distFromEdge = min(coord + 0.5f, dim - coord - 0.5f);
    fade = min(fade, smoothstep(0.0f, fadeWidth, distFromEdge));
  }
  return fade;
}

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uvec3 threadID : SV_DispatchThreadID)
{
  if (any(greaterThanEqual(threadID, cb.gridDimension)))
  {
    return;
  }

  // Obstacle cells receive zero output
  if (ObstacleInput[threadID] > 0.5f)
  {
    if (pushConst.advectVelocity == 0u)
    {
      Density4Output[threadID] = vec4(0.0f);
    }
    else
    {
      VelocityOutput[threadID] = vec4(0.0f);
    }
    return;
  }

  const float dt = cb.deltaTimeSecs;

  // World-space position of this cell's center
  const vec3 cellWorldPos = particleVolumeGridToWorld(threadID, cb);

  // Sample velocity at this cell and backtrace.
  // Semi-Lagrangian advection is unconditionally stable for any timestep,
  // but we clamp the displacement to prevent sampling from far outside the
  // grid.  NvFlow uses +/-15 cells; we use +/-10 for a tighter bound.
  const vec3 velocity = particleVolumeSampleVelocity(VelocityInput, LinearSampler, cellWorldPos, cb);
  const vec3 cellSize = (cb.aabbMax - cb.aabbMin) / vec3(cb.gridDimension);
  static const float kMaxDisplacementCells = 10.0f;
  const vec3 maxDisplacement = cellSize * kMaxDisplacementCells;
  vec3 displacement = velocity * dt;
  displacement = clamp(displacement, -maxDisplacement, maxDisplacement);
  vec3 sourcePos = cellWorldPos - displacement;

  // Clamp source position to the volume AABB (prevents out-of-texture reads)
  sourcePos = clamp(sourcePos, cb.aabbMin, cb.aabbMax);

  // Boundary fade: smoothly attenuate all quantities near grid edges.
  // This creates open-boundary behavior -- density and velocity dissipate as
  // they approach the edges rather than recirculating.
  const float boundaryFade = particleVolumeBoundaryFade(threadID, cb);

  if (pushConst.advectVelocity == 0u)
  {
    // ================================================================
    // DIAGNOSTIC: Write a known test pattern when debugMode >= 1.
    // This bypasses normal advection to verify the resolve shader can
    // actually read the density4 texture.  If debug mode 1 shows
    // a red sphere, the binding/barrier chain works and the issue is
    // in the simulation pipeline (splat/combustion).  If it still
    // shows nothing, the resolve shader cannot see this texture.
    // ================================================================
    if (cb.debugMode >= 1u)
    {
      // Gradient sphere: max at grid center, fading to zero at edges.
      vec3 uvw = vec3(threadID) / vec3(cb.gridDimension);
      float dist = length(uvw - 0.5f) * 2.0f; // 0 at center, 1 at corners
      float val = saturate(1.0f - dist);       // 1 at center, 0 at edge
      // Write temp=val, fuel=val*0.5, burn=val*0.1, smoke=val*0.3
      Density4Output[threadID] = vec4(val, val * 0.5f, val * 0.1f, val * 0.3f);
      return;
    }

    // ================================================================
    // DENSITY4 ADVECTION + COMBUSTION + NvFlow DAMPING
    // ================================================================

    // 1. Semi-Lagrangian backtrace: sample previous density4
    vec4 d = particleVolumeSampleDensity4(Density4Input, LinearSampler, sourcePos, cb);
    float temp  = d.x;
    float fuel  = d.y;
    float burn  = 0.0f; // burn is transient -- reset each frame
    float smoke = d.w;

    // 2. MacCormack correction (on temp, fuel, smoke -- not burn)
    if (pushConst.useMacCormack == 1u)
    {
      // Reverse advection: sample velocity at backtrace point, trace forward
      vec3 velAtSource = particleVolumeSampleVelocity(VelocityInput, LinearSampler, sourcePos, cb);
      vec3 reverseDisplacement = velAtSource * dt;
      reverseDisplacement = clamp(reverseDisplacement, -maxDisplacement, maxDisplacement);
      vec3 reversePos = sourcePos + reverseDisplacement;
      reversePos = clamp(reversePos, cb.aabbMin, cb.aabbMax);

      // Sample reverse-advected values (should approximate original cell values)
      vec4 dRev = particleVolumeSampleDensity4(Density4Input, LinearSampler, reversePos, cb);

      // Sample original values at cell center
      vec4 dOrig = particleVolumeSampleDensity4(Density4Input, LinearSampler, cellWorldPos, cb);

      // MacCormack error correction: half the difference between original
      // and reverse-advected values estimates the interpolation error.
      // Apply to temp, fuel, smoke (not burn -- burn is reset to 0).
      temp  += 0.5f * (dOrig.x - dRev.x);
      fuel  += 0.5f * (dOrig.y - dRev.y);
      smoke += 0.5f * (dOrig.w - dRev.w);

      // Clamp against min/max in neighborhood at sourcePos for stability.
      // Sample the 8 corners of the cell containing the backtrace point to
      // prevent the correction from introducing new extrema (overshooting).
      vec3 sourceGrid = particleVolumeWorldToGrid(sourcePos, cb);
      ivec3 baseCell = ivec3(floor(sourceGrid));
      vec4 dMin = vec4(1e20f);
      vec4 dMax = vec4(-1e20f);
      for (int dz = 0; dz <= 1; dz++)
      for (int dy = 0; dy <= 1; dy++)
      for (int dx = 0; dx <= 1; dx++)
      {
        ivec3 nc = clamp(baseCell + ivec3(dx, dy, dz), ivec3(0), ivec3(cb.gridDimension) - 1);
        vec4 nd = Density4Input[nc];
        dMin = min(dMin, nd);
        dMax = max(dMax, nd);
      }
      temp  = clamp(temp,  dMin.x, dMax.x);
      fuel  = clamp(fuel,  dMin.y, dMax.y);
      smoke = clamp(smoke, dMin.w, dMax.w);
    }

    // 3. Combustion (Nguyen, Fedkiw & Jensen 2002 / NvFlow model)
    if (temp >= cb.ignitionTemp && fuel > 0.0f)
    {
      burn = cb.burnPerTemp * clamp(temp, 0.0f, 1.0f) * dt;
      burn = min(burn, cb.fuelPerBurn * fuel);
      fuel  -= cb.fuelPerBurn * burn;
      temp  += cb.tempPerBurn * burn;
      smoke += cb.smokePerBurn * burn;
    }

    // Exponential cooling
    temp -= cb.coolingRate * dt * temp;
    temp = clamp(temp, 0.0f, 1.0f);

    // 4. NvFlow damping (per channel)
    //   correction = sign(val) * min(|val|, max(|val * dampingRate|, |dt * fadeRate|))
    //   val -= correction
    vec4 val = vec4(temp, fuel, burn, smoke);
    vec4 correction = sign(val) * min(abs(val), max(abs(val * cb.dampingRate), abs(dt * cb.fadeRate)));
    val -= correction;

    // 5. Boundary fade
    val *= boundaryFade;

    Density4Output[threadID] = val;
  }
  else
  {
    // ================================================================
    // VELOCITY ADVECTION + NvFlow DAMPING + FORCES (buoyancy/wind/vorticity)
    // ================================================================

    // 1. Semi-Lagrangian backtrace: sample previous velocity
    vec3 vel = particleVolumeSampleVelocity(VelocityInput, LinearSampler, sourcePos, cb);

    // 2. MacCormack correction for velocity field
    if (pushConst.useMacCormack == 1u)
    {
      // Reverse advection: sample velocity at backtrace point, trace forward
      vec3 velAtSource = particleVolumeSampleVelocity(VelocityInput, LinearSampler, sourcePos, cb);
      vec3 reverseDisplacement = velAtSource * dt;
      reverseDisplacement = clamp(reverseDisplacement, -maxDisplacement, maxDisplacement);
      vec3 reversePos = sourcePos + reverseDisplacement;
      reversePos = clamp(reversePos, cb.aabbMin, cb.aabbMax);

      // Sample reverse-advected velocity (should approximate original cell value)
      vec3 velRev = particleVolumeSampleVelocity(VelocityInput, LinearSampler, reversePos, cb);

      // Sample original velocity at cell center
      vec3 velOrig = particleVolumeSampleVelocity(VelocityInput, LinearSampler, cellWorldPos, cb);

      // MacCormack error correction
      vel += 0.5f * (velOrig - velRev);

      // Clamp against min/max in neighborhood at sourcePos for stability
      vec3 sourceGrid = particleVolumeWorldToGrid(sourcePos, cb);
      ivec3 baseCell = ivec3(floor(sourceGrid));
      vec3 vMin = vec3(1e20f);
      vec3 vMax = vec3(-1e20f);
      for (int dz = 0; dz <= 1; dz++)
      for (int dy = 0; dy <= 1; dy++)
      for (int dx = 0; dx <= 1; dx++)
      {
        ivec3 nc = clamp(baseCell + ivec3(dx, dy, dz), ivec3(0), ivec3(cb.gridDimension) - 1);
        vec3 nv = VelocityInput[nc].xyz;
        vMin = min(vMin, nv);
        vMax = max(vMax, nv);
      }
      vel = clamp(vel, vMin, vMax);
    }

    // 3. NvFlow velocity damping
    //   correction = sign(vel) * min(|vel|, max(|vel * velDampingRate|, |dt * velFadeRate|))
    //   vel -= correction
    vec3 corr = sign(vel) * min(abs(vel), max(abs(vel * cb.velDampingRate.xyz), abs(dt * cb.velFadeRate.xyz)));
    vel -= corr;

    // 4. Buoyancy (read post-combustion density4 from current frame via Texture3D)
    //   The density4 pass has already written this frame's density4; it is bound
    //   read-only here as Density4Input.
    vec4 d = Density4Input[threadID];
    // NvFlow: vel -= buoyancy * gravity, where gravity points DOWN.
    // Our gravity = -upDirection * gravityMagnitude (points down).
    // Temperature buoyancy lifts UP (opposes gravity), smoke buoyancy pulls DOWN (with gravity).
    vec3 gravity = -cb.upDirection * cb.gravityMagnitude;
    vel -= dt * cb.buoyancyPerTemp * d.x * gravity;
    vel -= dt * cb.buoyancyPerSmoke * clamp(d.w, 0.0f, cb.buoyancyMaxSmoke) * gravity;

    // 5. Wind: uniform directional force
    vel += cb.windDirection * dt;

    // 6. Vorticity confinement
    if (cb.vorticityConfinement > 0.0f)
    {
      const ivec3 cell = ivec3(threadID);
      const vec3 curl = particleVolumeCalcCurl(VelocityInput, cell, cb);

      // Compute curl magnitude at the six face-neighbours for gradient estimation
      const float cxp = length(particleVolumeCalcCurl(VelocityInput, cell + ivec3(1, 0, 0), cb));
      const float cxn = length(particleVolumeCalcCurl(VelocityInput, cell - ivec3(1, 0, 0), cb));
      const float cyp = length(particleVolumeCalcCurl(VelocityInput, cell + ivec3(0, 1, 0), cb));
      const float cyn = length(particleVolumeCalcCurl(VelocityInput, cell - ivec3(0, 1, 0), cb));
      const float czp = length(particleVolumeCalcCurl(VelocityInput, cell + ivec3(0, 0, 1), cb));
      const float czn = length(particleVolumeCalcCurl(VelocityInput, cell - ivec3(0, 0, 1), cb));

      // Central-difference gradient of curl magnitude
      const vec3 grad = vec3(cxp - cxn, cyp - cyn, czp - czn) * 0.5f;

      // Normalise to obtain confinement direction; guard against zero-length gradient
      const float gradLen = length(grad);
      if (gradLen > 1e-6f)
      {
        const vec3 N = grad / gradLen;
        vel += cross(N, curl) * cb.vorticityConfinement * dt;
      }
    }

    // 7. Divergence offset from combustion (stored in velocity.w for pressure solver)
    float divOffset = cb.divergencePerBurn * d.z;

    // 8. Boundary fade
    vel *= boundaryFade;

    VelocityOutput[threadID] = vec4(vel, divOffset);
  }
}
