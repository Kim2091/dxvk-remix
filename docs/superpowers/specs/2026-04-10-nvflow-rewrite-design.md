# NvFlow-Faithful Volumetric Smoke/Fire Rewrite

## Overview

Rewrite the volumetric smoke/fire simulation to closely follow NvFlow's architecture, using normalized [0,1] value ranges, blend-toward-target emission, NvFlow's exact damping formula, and colormap-based rendering. Reuses existing GPU infrastructure (textures, TLAS proxy, dispatch framework, pressure solver, obstacle pass).

## Problem Statement

The current implementation mixes physical units (Kelvin temperatures, world-space densities) with a system designed for normalized ranges. Additive emission accumulates without bound, requiring ad-hoc caps. Competing damping mechanisms fight each other. The result is unstable — simulation blows out or produces no visible output depending on parameter tuning.

## Design

### 1. Value Model & Data Layout

**Packed float4 density texture (R16G16B16A16_SFLOAT):**
| Channel | Field | Range | Meaning |
|---------|-------|-------|---------|
| .x | temperature | [0, 1] | 0 = ambient, 1 = max burn |
| .y | fuel | [0, 1] | Available fuel at this cell |
| .z | burn | [0, 1] | Transient burn rate this frame |
| .w | smoke | [0, ∞) | Accumulated smoke opacity |

**Velocity texture (R16G16B16A16_SFLOAT):**
| Channel | Field | Range | Meaning |
|---------|-------|-------|---------|
| .xyz | velocity | unbounded | World-space units/sec |
| .w | divergence | unbounded | Combustion expansion term for pressure solver |

**Resource reduction:** Replaces 6 separate scalar 3D textures (density, prevDensity, temperature, prevTemperature, fuel, prevFuel) with 2 packed float4 textures (density4, prevDensity4). Saves memory and simplifies bindings.

**Bytes per cell:** density4 (8) + prevDensity4 (8) + velocity (8) + prevVelocity (8) + obstacle (1) + pressure×2 (4) = **37 bytes** (vs 33 currently, slight increase due to burn channel, but fewer textures to manage).

### 2. Emission (Splat) Model

**Blend-toward-target coupling** (NvFlow EmitterBoxCS pattern):

```hlsl
float4 coupleRate = saturate(cb.emitterCoupleRate * dt * weight);
value += coupleRate * (targetValue - value);
```

Self-limiting: `coupleRate` clamped to [0,1], so values can never overshoot the target. No per-cell caps needed.

**Target values from particles:**
- temperature: `fuelAmount` (normalized, e.g. 0.8)
- fuel: `fuelAmount`
- smoke: 0.0 (combustion creates smoke, not direct injection)
- burn: 0.0 (computed by combustion)
- velocity: `particle.velocity * fluidCouplingStrength`

**Coupling rate:** Single `emitterCoupleRate` parameter, default 3.0/sec.

**Life gating:** Only particles with lifeFraction >= 0.7 (first 30% of life) contribute. Unchanged from current system.

### 3. Advection & Combustion

**Single combined density pass** (matching NvFlow AdvectionDensity2CS):

1. Semi-Lagrangian backtrace with displacement clamping (±10 cells)
2. MacCormack correction with neighbor-extrema clamping
3. Combustion (NvFlow formula):
   ```
   if temp >= ignitionTemp and fuel > 0:
     burn = burnPerTemp * clamp(temp, 0, 1) * dt
     burn = min(burn, burnPerFuel * fuel)
     fuel -= fuelPerBurn * burn
     temp += tempPerBurn * burn
     smoke += smokePerBurn * burn
     temp -= coolingRate * dt * temp
     temp = clamp(temp, 0, 1)
   ```
4. NvFlow damping formula:
   ```
   correction = sign(val) * min(abs(val), max(abs(val * dampingRate), abs(dt * fadeRate)))
   val -= correction
   ```
   `dampingRate` computed on CPU: `1 - pow(1 - damping_per_sec, dt)`
5. Boundary fade (smoothstep at grid edges, ~10% zone width)

**Velocity advection pass:** Same pipeline minus combustion, plus buoyancy force application.

**Default damping/fade values (NvFlow defaults):**
| Channel | Damping (per sec) | Fade (per sec) |
|---------|-------------------|----------------|
| velocity | 0.01 | 1.0 |
| temperature | 0.0 | 0.0 |
| fuel | 0.0 | 0.0 |
| burn | 0.01 | 1.0 |
| smoke | 0.30 | 0.65 |

### 4. Forces & Pressure

**Buoyancy** (NvFlow formula, normalized):
```hlsl
velocity.xyz -= dt * buoyancyPerTemp * temp * gravity;
velocity.xyz -= dt * buoyancyPerSmoke * clamp(smoke, 0, buoyancyMaxSmoke) * gravity;
```

`gravity` = scene up direction × magnitude (default 100, matching NvFlow). Temperature [0,1] directly drives buoyancy — no ambient subtraction or normalization needed.

**Combustion divergence:** `velocity.w = divergencePerBurn * burn` — feeds into pressure solver.

**Pressure solver:** Keep existing Jacobi+SOR (ω=1.4). One change: include velocity.w in divergence computation:
```hlsl
divergence = central_diff(velocity.xyz) + velocity_at_cell.w;
```

Default `divergencePerBurn = 0.0` (disabled, no behavior change until enabled).

Wind and vorticity confinement unchanged.

### 5. Rendering

**Colormap-based** (NvFlow NvFlowRayMarch pattern):

**Default colormap** generated at startup (256×1 RGBA):
- u=0.0: transparent black
- u=0.1: dark red
- u=0.3: bright red-orange
- u=0.5: orange-yellow
- u=0.8: yellow-white
- u=1.0: bright white
- Alpha: 0→1 ramp over [0, 0.3], then 1.0

**Ray-march per sample:**
```hlsl
float4 d = densityTexture.SampleLevel(sampler, uvw, 0);
float4 color = colormapTexture.SampleLevel(sampler, float2(d.x, 0.5), 0);
color.rgb *= (1.0 + shadowFactor * (d.z - 1.0));  // burn modulates brightness
color.a *= d.w;                                     // smoke controls opacity
color.rgb *= colorScale;
color.a *= alphaScale;

float sigma_t = d.w * absorptionCrossSection;       // smoke drives extinction
float stepT = exp(-sigma_t * stepSize);
radiance += transmittance * color.a * color.rgb * stepSize;
transmittance *= stepT;
```

**Self-shadowing:** Keep existing 6-step secondary march, reads smoke channel (d.w) instead of separate density.

**Ambient scattering:** `kAmbientColor * smoke` for visibility in dark scenes.

**Froxel injection & light extraction:** Updated to read packed float4, use smoke for extinction and colormap(temp) for emission.

### 6. Constants Struct

Replace current mixed-unit `ParticleVolumeConstants` with NvFlow-aligned parameters:

```c
struct ParticleVolumeConstants {
  mat4 worldToVolume;
  mat4 volumeToWorld;
  mat4 prevWorldToProjection;

  vec3 aabbMin;              float deltaTime;
  vec3 aabbMax;              float absoluteTime;
  uvec3 gridDimension;       uint frameIdx;

  // Combustion
  float ignitionTemp;        // [0,1] threshold
  float burnPerTemp;         // burn rate per excess temp (default 4.0)
  float fuelPerBurn;         // fuel consumed per burn (default 0.25)
  float tempPerBurn;         // temp generated per burn (default 5.0)
  float smokePerBurn;        // smoke generated per burn (default 3.0)
  float coolingRate;         // exponential cooling rate (default 1.5)
  float divergencePerBurn;   // expansion in pressure solver (default 0.0)
  float emitterCoupleRate;   // emission blend rate per sec (default 3.0)

  // Damping (CPU-computed per-frame from per-second values)
  vec4 dampingRate;          // per-channel: 1 - pow(1 - damping, dt)
  vec4 fadeRate;             // per-channel: absolute fade per second

  // Forces
  float buoyancyPerTemp;     // default 2.0
  float buoyancyPerSmoke;    // default 0.0
  float buoyancyMaxSmoke;    // default 1.0
  float gravityMagnitude;    // default 100.0
  vec3 upDirection;          float vorticityConfinement;
  vec3 windDirection;        float fluidCouplingStrength;

  // Rendering
  float absorptionCrossSection;
  float colorScale;          // emission brightness multiplier
  float alphaScale;          // smoke opacity multiplier
  float shadowFactor;        // burn-to-brightness modulation

  // Pipeline
  uint pressureIterations;
  uint particleCount;
  uint renderingWidth;
  uint renderingHeight;
  vec3 cameraPosition;       float maxTimeToLive;
  float sceneScale;
  float pad0, pad1, pad2;
};
```

### 7. Scope Summary

**Rewrite (5 shaders):**
- `particle_volume_advect.comp.slang` — packed float4, NvFlow combustion+damping
- `particle_volume_splat.comp.slang` — blend-toward-target coupling
- `particle_volume_forces.comp.slang` — normalized buoyancy, divergence offset
- `particle_volume_froxel_inject.comp.slang` — read packed float4
- `particle_volume_light_extract.comp.slang` — read packed float4

**Modify (3 shaders):**
- `particle_volume_pressure.comp.slang` — add velocity.w divergence offset
- `blackbody_lut.comp.slang` → `particle_volume_colormap.comp.slang` — fire gradient
- `resolve.slangh` ray-march — colormap rendering, smoke-based extinction

**Keep as-is (2 shaders):**
- `particle_volume_obstacle.comp.slang`
- `particle_volume_project.comp.slang`

**C++ changes:**
- `rtx_particle_volume.h/cpp` — packed float4 textures, updated bindings
- `rtx_particle_system.h/cpp` — NvFlow-style RTX_OPTIONs, ImGui, constants setup
- `particle_volume_common.h` — new constants struct
- `particle_volume_helpers.slangh` — float4 sampling helpers
- `particle_volume_binding_indices.h` — simplified (fewer textures)
- Test scene parameter updates

**No changes:**
- Particle spawn/simulation
- TLAS proxy registration
- Resolution LOD / hysteresis
- Common raytracing bindings
