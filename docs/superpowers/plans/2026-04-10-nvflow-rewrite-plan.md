# NvFlow-Faithful Rewrite — Implementation Plan

Spec: `docs/superpowers/specs/2026-04-10-nvflow-rewrite-design.md`

**Spec deviations (intentional):**
- Spec lists forces shader as "Rewrite" — plan DELETES it instead, folding buoyancy/wind/vorticity into the velocity advection pass. This matches NvFlow's `AdvectionCombustionVelocity` architecture.
- Spec constants struct is missing `velDampingRate`/`velFadeRate` vec4 fields — plan adds them (velocity needs separate damping/fade from density channels).
- Spec colormap alpha ramp is [0, 0.3] — plan uses [0, 0.15] for a tighter fire threshold.

## Dependency Graph

```
Step 1: Constants & data layout    (no deps — foundation everything else reads)
Step 2: GPU resources              (depends on 1 — allocate packed textures)
Step 3: Colormap shader            (depends on 1 — replaces blackbody LUT)
Step 4: Splat shader               (depends on 1, 2 — first shader to write density4)
Step 5: Advection shader           (depends on 1, 2 — reads/writes density4, combustion+damping+forces)
Step 6: Forces shader → DELETE     (depends on nothing — just delete the file)
Step 7: Pressure shader            (depends on 1 — add divergence offset, one-line change)
Step 8: Ray-march + bindings       (depends on 1, 3 — colormap rendering, common_bindings update)
Step 9: Froxel + light extract     (depends on 1 — read packed float4)
Step 10: C++ dispatch wiring       (depends on 2-9 — bind new textures, update dispatch)
Step 11: ImGui + RTX options       (depends on 1 — new parameter set)
Step 12: Test scene                (depends on 11 — update descriptor values)
Step 13: Build + smoke test        (depends on all)
```

Steps 3-9 are largely independent of each other (all depend on 1-2). They can be parallelized across subagents.

---

## Step 1: Constants & Data Layout

**Files:**
- `src/dxvk/shaders/rtx/pass/particles/particle_volume_common.h`
- `src/dxvk/shaders/rtx/pass/particles/particle_volume_binding_indices.h`

**Actions:**

Replace `ParticleVolumeConstants` struct entirely:

```c
struct ParticleVolumeConstants
{
  mat4 worldToVolume;
  mat4 volumeToWorld;
  mat4 prevWorldToProjection;

  vec3 aabbMin;               float deltaTimeSecs;    // keep existing name to avoid renaming in all shaders
  vec3 aabbMax;               float absoluteTimeSecs; // keep existing name
  uvec3 gridDimension;        uint frameIdx;

  // Combustion (Nguyen/Fedkiw/NvFlow)
  float ignitionTemp;         // [0,1] threshold for combustion (default 0.05)
  float burnPerTemp;          // burn rate per excess temp (default 4.0)
  float fuelPerBurn;          // fuel consumed per burn (default 0.25) — clamps burn: burn = min(burn, fuelPerBurn * fuel)
  float tempPerBurn;          // temp generated per burn (default 5.0)
  float smokePerBurn;         // smoke generated per burn (default 3.0)
  float coolingRate;          // exponential cooling (default 1.5)
  float divergencePerBurn;    // pressure expansion (default 0.0)
  float emitterCoupleRate;    // emission blend rate /sec (default 3.0)

  // Damping (CPU-computed per frame)
  vec4 dampingRate;           // 1 - pow(1 - damping, dt), per channel (temp/fuel/burn/smoke)
  vec4 fadeRate;              // absolute fade /sec, per channel
  vec4 velDampingRate;        // velocity damping (xyz used, w unused)
  vec4 velFadeRate;           // velocity fade (xyz used, w unused)

  // Forces
  float buoyancyPerTemp;      // default 2.0
  float buoyancyPerSmoke;     // default 0.0
  float buoyancyMaxSmoke;     // default 1.0
  float gravityMagnitude;     // default 100.0
  vec3 upDirection;           float vorticityConfinement;
  vec3 windDirection;         float fluidCouplingStrength;

  // Rendering
  float absorptionCrossSection; // smoke extinction coefficient
  float colorScale;           // emission brightness multiplier
  float alphaScale;           // smoke opacity multiplier
  float shadowFactor;         // burn brightness modulation

  // Pipeline
  uint pressureIterations;
  uint particleCount;
  uint renderingWidth;
  uint renderingHeight;
  vec3 cameraPosition;        float maxTimeToLive;
  float sceneScale;
  float fuelAmount;           // normalized fuel injection target [0,1]
  float pad0;
  float pad1;
};
```

Update binding indices — remove separate FUEL_INPUT/FUEL_OUTPUT (fuel is now packed in density4). Remaining bindings:

```c
// Inputs
#define PARTICLE_VOLUME_BINDING_CONSTANTS                  0
#define PARTICLE_VOLUME_BINDING_DENSITY4_INPUT             1   // float4: temp/fuel/burn/smoke
#define PARTICLE_VOLUME_BINDING_VELOCITY_INPUT             3
#define PARTICLE_VOLUME_BINDING_OBSTACLE_INPUT             4
#define PARTICLE_VOLUME_BINDING_PRESSURE_INPUT             5

// Outputs
#define PARTICLE_VOLUME_BINDING_DENSITY4_OUTPUT            10  // float4: temp/fuel/burn/smoke
#define PARTICLE_VOLUME_BINDING_VELOCITY_OUTPUT            12
#define PARTICLE_VOLUME_BINDING_OBSTACLE_OUTPUT            13
#define PARTICLE_VOLUME_BINDING_PRESSURE_OUTPUT            14

// Particle buffer
#define PARTICLE_VOLUME_BINDING_PARTICLES_INPUT            20
#define PARTICLE_VOLUME_BINDING_PREV_WORLD_POSITION_INPUT  21

// Colormap + sampler
#define PARTICLE_VOLUME_BINDING_COLORMAP_INPUT             23
#define PARTICLE_VOLUME_BINDING_LIGHT_OUTPUT               24
#define PARTICLE_VOLUME_BINDING_FROXEL_DENSITY_OUTPUT      25
#define PARTICLE_VOLUME_BINDING_FROXEL_EMISSION_OUTPUT     26
#define PARTICLE_VOLUME_BINDING_LINEAR_SAMPLER             30

// REMOVED (no longer used):
// PARTICLE_VOLUME_BINDING_TEMPERATURE_INPUT (2) — packed into density4
// PARTICLE_VOLUME_BINDING_PREV_VELOCITY_INPUT (6) — velocity pass uses VELOCITY_INPUT
// PARTICLE_VOLUME_BINDING_FUEL_INPUT (7) — packed into density4
// PARTICLE_VOLUME_BINDING_TEMPERATURE_OUTPUT (11) — packed into density4
// PARTICLE_VOLUME_BINDING_FUEL_OUTPUT (15) — packed into density4
// PARTICLE_VOLUME_BINDING_DEPTH_INPUT (22) — unused
// PARTICLE_VOLUME_BINDING_COMPOSITE_WORLD_POS_INPUT (31) — composite pass removed
// PARTICLE_VOLUME_BINDING_COMPOSITE_COLOR_INOUT (32) — composite pass removed
```

Update helpers in `particle_volume_helpers.slangh`:

```hlsl
// Sample packed density4 at world position
vec4 particleVolumeSampleDensity4(Texture3D<vec4> tex, SamplerState s, vec3 worldPos, ParticleVolumeConstants cb)
{
  vec3 uvw = (worldPos - cb.aabbMin) / (cb.aabbMax - cb.aabbMin);
  return tex.SampleLevel(s, uvw, 0);
}

// Sample velocity at world position
vec3 particleVolumeSampleVelocity(Texture3D<vec4> tex, SamplerState s, vec3 worldPos, ParticleVolumeConstants cb)
{
  vec3 uvw = (worldPos - cb.aabbMin) / (cb.aabbMax - cb.aabbMin);
  return tex.SampleLevel(s, uvw, 0).xyz;
}
```

Remove old `particleVolumeSampleDensity`, `particleVolumeSampleTemperature`. Keep `particleVolumeGridToWorld`, `particleVolumeWorldToGrid`, `particleVolumeIsInBounds`, `particleVolumeCalcSplatWeights`, `particleVolumeCalcCurl`, `particleVolumeBoundaryFade`.

**Validation:** Struct size must be 16-byte aligned. Count manually before implementing.

---

## Step 2: GPU Resources

**Files:**
- `src/dxvk/rtx_render/rtx_particle_volume.h`
- `src/dxvk/rtx_render/rtx_particle_volume.cpp`

**Actions:**

Replace texture members:

```cpp
// REMOVE:
Resources::Resource m_density;        // R16_SFLOAT
Resources::Resource m_prevDensity;
Resources::Resource m_temperature;
Resources::Resource m_prevTemperature;
Resources::Resource m_fuel;
Resources::Resource m_prevFuel;

// ADD:
Resources::Resource m_density4;       // R16G16B16A16_SFLOAT — packed temp/fuel/burn/smoke
Resources::Resource m_prevDensity4;   // R16G16B16A16_SFLOAT — previous frame snapshot
```

Keep: `m_velocity`, `m_prevVelocity`, `m_obstacle`, `m_pressure[2]`, `m_cb`.

Update `allocate()`:
- Create `m_density4` and `m_prevDensity4` as `VK_FORMAT_R16G16B16A16_SFLOAT`
- Remove density, prevDensity, temperature, prevTemperature, fuel, prevFuel creation

Update `release()`: reset the new resources, remove old ones.

Update `kBytesPerCell`:
- density4: 8 + prevDensity4: 8 + velocity: 8 + prevVelocity: 8 + obstacle: 1 + pressure×2: 4 = **37 bytes**

Update accessors:
```cpp
const Resources::Resource& density4Texture() const { return m_density4; }
// Remove: densityTexture(), temperatureTexture(), fuelTexture()
```

Update snapshot copy in `simulateFluid()`: copy `m_density4 → m_prevDensity4` (one copy instead of three).

---

## Step 3: Colormap Shader

**Files:**
- `src/dxvk/shaders/rtx/pass/particles/blackbody_lut.comp.slang` → rename to `particle_volume_colormap.comp.slang`
- Update meson.build shader list if needed

**Actions:**

Rewrite to generate a 256×1 RGBA fire gradient:

```hlsl
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint threadID : SV_DispatchThreadID)
{
  float u = float(threadID) / 255.0;

  // Fire gradient: black → dark red → orange → yellow → white
  vec3 color;
  if (u < 0.1)
    color = mix(vec3(0.0), vec3(0.3, 0.0, 0.0), u / 0.1);
  else if (u < 0.3)
    color = mix(vec3(0.3, 0.0, 0.0), vec3(1.0, 0.3, 0.0), (u - 0.1) / 0.2);
  else if (u < 0.5)
    color = mix(vec3(1.0, 0.3, 0.0), vec3(1.0, 0.7, 0.1), (u - 0.3) / 0.2);
  else if (u < 0.8)
    color = mix(vec3(1.0, 0.7, 0.1), vec3(1.0, 0.95, 0.7), (u - 0.5) / 0.3);
  else
    color = mix(vec3(1.0, 0.95, 0.7), vec3(1.0, 1.0, 1.0), (u - 0.8) / 0.2);

  // Alpha: ramp from 0 to 1 over [0, 0.15], then 1.0
  float alpha = saturate(u / 0.15);

  ColormapOutput[uint2(threadID, 0)] = vec4(color, alpha);
}
```

This generates a plausible fire palette. The exact gradient can be tuned later.

---

## Step 4: Splat Shader

**Files:**
- `src/dxvk/shaders/rtx/pass/particles/particle_volume_splat.comp.slang`

**Actions:**

Complete rewrite. New version:

**Bindings:** DENSITY4_OUTPUT (RWTexture3D<vec4>), VELOCITY_OUTPUT (RWTexture3D<vec4>), PARTICLES_INPUT.

**Per-particle logic:**
1. Skip dead/sleeping/out-of-bounds particles
2. Skip particles with lifeFraction < 0.7
3. Compute trilinear splat weights (unchanged)
4. For each of 8 neighbor cells:
   ```hlsl
   float rate = saturate(cb.emitterCoupleRate * cb.deltaTimeSecs * weight);

   // Density4: blend toward target
   vec4 existing = Density4Output[coord];
   vec4 target = vec4(cb.fuelAmount, cb.fuelAmount, 0.0, 0.0); // temp, fuel, burn=0, smoke=0
   Density4Output[coord] = mix(existing, target, rate);

   // Velocity: blend toward particle velocity
   vec4 existingVel = VelocityOutput[coord];
   vec3 velTarget = particle.velocity * cb.fluidCouplingStrength;
   existingVel.xyz = mix(existingVel.xyz, velTarget, rate);
   VelocityOutput[coord] = existingVel;
   ```

Note: `mix(a, b, t) = a + t * (b - a)` = NvFlow's `value += rate * (target - value)`.

---

## Step 5: Advection Shader

**Files:**
- `src/dxvk/shaders/rtx/pass/particles/particle_volume_advect.comp.slang`

**Actions:**

Complete rewrite. Two modes via push constant:

**advectVelocity = 0 (density4 advection + combustion):**

```hlsl
// 1. Backtrace
vec3 sourcePos = cellWorldPos - clamp(velocity * dt, -maxDisp, maxDisp);
sourcePos = clamp(sourcePos, cb.aabbMin, cb.aabbMax);

// 2. Sample previous density4
vec4 d = particleVolumeSampleDensity4(PrevDensity4Input, LinearSampler, sourcePos, cb);
float temp  = d.x;
float fuel  = d.y;
float burn  = 0.0; // burn is transient, reset each frame
float smoke = d.w;

// 3. MacCormack correction (on temp, fuel, smoke — not burn)
if (pushConst.useMacCormack == 1u)
{
  // ... reverse advection, error correction, neighbor clamping
  // Same pattern as current MacCormack but operates on vec4
}

// 4. Combustion
if (temp >= cb.ignitionTemp && fuel > 0.0)
{
  burn = cb.burnPerTemp * clamp(temp, 0.0, 1.0) * dt;
  burn = min(burn, cb.fuelPerBurn * fuel);
  fuel -= cb.fuelPerBurn * burn;
  temp += cb.tempPerBurn * burn;
  smoke += cb.smokePerBurn * burn;
}
temp -= cb.coolingRate * dt * temp;
temp = clamp(temp, 0.0, 1.0);

// 5. NvFlow damping (per channel)
vec4 val = vec4(temp, fuel, burn, smoke);
vec4 correction = sign(val) * min(abs(val), max(abs(val * cb.dampingRate), abs(dt * cb.fadeRate)));
val -= correction;

// 6. Boundary fade
val *= particleVolumeBoundaryFade(threadID, cb);

Density4Output[threadID] = val;
```

**advectVelocity = 1 (velocity advection + forces):**

The velocity pass needs to READ density4 (post-combustion) for buoyancy. To avoid a read-write hazard, bind density4 as a read-only `Texture3D<vec4>` using `DENSITY4_INPUT` (not the RWTexture3D output slot). The C++ dispatch binds `m_density4` (current frame, written by density advection pass) to `DENSITY4_INPUT` for this pass.

```hlsl
// 1. Backtrace + MacCormack on velocity
vec3 vel = ... // semi-Lagrangian + MacCormack

// 2. NvFlow velocity damping
vec3 corr = sign(vel) * min(abs(vel), max(abs(vel * cb.velDampingRate.xyz), abs(dt * cb.velFadeRate.xyz)));
vel -= corr;

// 3. Buoyancy (read density4 at this cell for temp/smoke — via read-only Texture3D)
vec4 d = Density4Input[threadID]; // post-combustion values from density pass, bound as Texture3D
vel -= dt * cb.buoyancyPerTemp * d.x * cb.upDirection * cb.gravityMagnitude;
vel -= dt * cb.buoyancyPerSmoke * clamp(d.w, 0.0, cb.buoyancyMaxSmoke) * cb.upDirection * cb.gravityMagnitude;

// 4. Wind
vel += cb.windDirection * dt;

// 5. Vorticity confinement (unchanged)

// 6. Divergence offset
float divOffset = cb.divergencePerBurn * d.z;

// 7. Boundary fade
vel *= particleVolumeBoundaryFade(threadID, cb);

VelocityOutput[threadID] = vec4(vel, divOffset);
```

**Important change from current system:** Buoyancy moves from the separate forces pass INTO the velocity advection pass. This matches NvFlow where combustion velocity effects (buoyancy, divergence) are computed in the advection correct pass, not a separate dispatch. This eliminates the forces pass entirely.

---

## Step 6: Forces Shader → DELETE

**Files:**
- `src/dxvk/shaders/rtx/pass/particles/particle_volume_forces.comp.slang`
- `src/dxvk/rtx_render/rtx_particle_volume.cpp` (remove dispatch)

**Actions:**

Delete `particle_volume_forces.comp.slang`. Buoyancy, wind, and vorticity are now computed inside the velocity advection pass (Step 5). This matches NvFlow's architecture where `AdvectionCombustionVelocity` handles both advection and force application in a single dispatch.

Remove `ParticleVolumeForces` ManagedShader class and its dispatch from `simulateFluid()`.

Remove `#include <rtx_shaders/particle_volume_forces.h>`.

---

## Step 7: Pressure Shader

**Files:**
- `src/dxvk/shaders/rtx/pass/particles/particle_volume_pressure.comp.slang`

**Actions:**

One-line change — add divergence offset from velocity.w:

```hlsl
// Current:
const float divergence = ((vxp.x - vxn.x) + (vyp.y - vyn.y) + (vzp.z - vzn.z)) * 0.5f;

// New:
const float baseDivergence = ((vxp.x - vxn.x) + (vyp.y - vyn.y) + (vzp.z - vzn.z)) * 0.5f;
const float divergence = baseDivergence + VelocityInput[threadID].w;
```

SOR (ω=1.4) stays as-is.

---

## Step 8: Ray-March + Common Bindings

**Files:**
- `src/dxvk/shaders/rtx/algorithm/resolve.slangh` — inline ray-march rewrite
- `src/dxvk/shaders/rtx/concept/particle_volume_proxy.h` — MemoryParticleVolume struct update
- `src/dxvk/shaders/rtx/pass/common_binding_indices.h` — replace density+temperature bindings with density4
- `src/dxvk/shaders/rtx/pass/common_bindings.slangh` — update texture declarations from Texture3D<float> to Texture3D<vec4>
- `src/dxvk/shaders/rtx/pass/particles/particle_volume_march.slangh` — DELETE (ray-march is inline in resolve)

**Actions:**

**common_binding_indices.h:** Replace the 8 separate density+temperature bindings (200-207) with 4 density4 bindings:
```c
#define BINDING_PARTICLE_VOLUME_DENSITY4_0  200  // Texture3D<vec4> — packed temp/fuel/burn/smoke
#define BINDING_PARTICLE_VOLUME_DENSITY4_1  201
#define BINDING_PARTICLE_VOLUME_DENSITY4_2  202
#define BINDING_PARTICLE_VOLUME_DENSITY4_3  203
// Remove: BINDING_PARTICLE_VOLUME_TEMPERATURE_0..3 (204-207)
#define BINDING_PARTICLE_VOLUME_COLORMAP    208  // renamed from BLACKBODY_LUT
#define BINDING_PARTICLE_VOLUME_LINEAR_SAMPLER 209  // unchanged
```
Update `COMMON_RAYTRACING_BINDINGS` macro to remove temperature entries.

**common_bindings.slangh:** Replace texture declarations:
```hlsl
// Remove:
Texture3D<float> particleVolumeDensity0..3;
Texture3D<float> particleVolumeTemperature0..3;
Texture2D<float4> particleVolumeBlackbodyLUT;
// Add:
Texture3D<vec4> particleVolumeDensity4_0..3;
Texture2D<vec4> particleVolumeColormap;
```

**particle_volume_march.slangh:** Delete this file. The ray-march is done inline in resolve.slangh. The standalone function used old separate-texture API and is no longer needed.

Replace the inline ray-march block in resolve.slangh (~lines 920-1010). The volume proxy struct (`MemoryParticleVolume`) needs a few field updates to carry new rendering params.

Update `particle_volume_proxy.h`:
```c
struct MemoryParticleVolume
{
  vec3 aabbMin;               float absorptionCrossSection;
  vec3 aabbMax;               float colorScale;
  uvec3 gridDimension;        uint volumeIndex;
  vec3 upDirection;           float alphaScale;
  float shadowFactor;         float pad0; float pad1; float pad2;
};
// 80 bytes (5 × vec4)
```

New ray-march inner loop:
```hlsl
// Sample packed density4
vec4 d = { 0, 0, 0, 0 };
switch (vol.volumeIndex) {
  case 0: d = particleVolumeDensity4_0.SampleLevel(sampler, uvw, 0); break;
  // ... cases 1-3
}

float smoke = d.w;
if (smoke > 0.001f)
{
  // Color from temperature colormap
  vec4 color = particleVolumeColormap.SampleLevel(sampler, vec2(d.x, 0.5f), 0);
  color.rgb *= (1.0f + vol.shadowFactor * (d.z - 1.0f));  // burn brightness
  color.a *= smoke;
  color.rgb *= vol.colorScale;
  color.a = min(color.a * vol.alphaScale, 1.0f);

  // Self-shadow (existing, reads smoke channel)
  if (d.x > 0.1f) { /* shadow march using smoke from density4 */ }

  // Extinction from smoke
  float sigma_t = smoke * vol.absorptionCrossSection;
  float stepT = exp(-sigma_t * stepSize);

  // Ambient scattering
  vec3 ambient = vec3(0.15f, 0.17f, 0.20f) * smoke;

  volRadiance += volTransmittance * (color.a * color.rgb + (1.0f - stepT) * ambient);
  volTransmittance *= stepT;
}
```

Update the density texture bindings — replace 4 separate `particleVolumeDensity0..3` textures with 4 `particleVolumeDensity4_0..3` (R16G16B16A16_SFLOAT). Update `rtx_context.cpp` binding code accordingly.

---

## Step 9: Froxel Injection + Light Extraction

**Files:**
- `src/dxvk/shaders/rtx/pass/particles/particle_volume_froxel_inject.comp.slang`
- `src/dxvk/shaders/rtx/pass/particles/particle_volume_light_extract.comp.slang`

**Actions:**

Both shaders currently read separate density and temperature textures. Update to read packed float4:

**Froxel injection:**
```hlsl
vec4 d = Density4Input.SampleLevel(LinearSampler, uvw, 0);
float smoke = d.w;
float temp = d.x;
if (smoke < 1e-5 && temp < 0.01) return; // skip negligible cells
float extinction = smoke * cb.absorptionCrossSection;
vec4 colormapColor = ColormapInput.SampleLevel(LinearSampler, vec2(temp, 0.5), 0);
vec3 emission = colormapColor.rgb * colormapColor.a * cb.colorScale;
// Write to froxel
```

**Light extraction:**
```hlsl
vec4 d = Density4Input[cell];
if (d.x > 0.1) // temp threshold for light contribution
{
  vec4 colormapColor = ColormapInput.SampleLevel(LinearSampler, vec2(d.x, 0.5), 0);
  vec3 emission = colormapColor.rgb * colormapColor.a * cb.colorScale;
  // Accumulate into reduction
}
```

---

## Step 10: C++ Dispatch Wiring

**Files:**
- `src/dxvk/rtx_render/rtx_particle_volume.cpp` — ManagedShader params, dispatch sequence, snapshot copies
- `src/dxvk/rtx_render/rtx_accel_manager.cpp` — update MemoryParticleVolume field assignments (absorptionCrossSection, colorScale, alphaScale, shadowFactor)
- `src/dxvk/rtx_render/rtx_context.cpp` — replace density+temperature texture bindings with density4, update dummy texture format to R16G16B16A16_SFLOAT, update resolve args constant buffer, rename ensureBlackbodyLUT() to ensureColormap()
- `src/dxvk/rtx_render/rtx_particle_system.h` — update `ActiveVolumeDescriptor` struct: replace `densityView`/`temperatureView` with `density4View`, replace `smokeDensity`/`emissionIntensityScale` with new rendering params
- `src/dxvk/rtx_render/rtx_particle_system.cpp` — update `getActiveVolumeDescriptors()` to populate new fields

**Actions:**

Update ManagedShader parameter blocks:
- `ParticleVolumeSplat`: DENSITY4_OUTPUT, VELOCITY_OUTPUT, PARTICLES_INPUT
- `ParticleVolumeAdvect`: DENSITY4_INPUT, VELOCITY_INPUT, OBSTACLE_INPUT, LINEAR_SAMPLER, DENSITY4_OUTPUT, VELOCITY_OUTPUT
- Remove `ParticleVolumeForces` entirely
- `ParticleVolumePressure`: VELOCITY_INPUT, PRESSURE_INPUT, OBSTACLE_INPUT, PRESSURE_OUTPUT (unchanged)
- `ParticleVolumeProject`: PRESSURE_INPUT, OBSTACLE_INPUT, VELOCITY_OUTPUT (unchanged)

Update `simulateFluid()` dispatch sequence:
1. Obstacle (unchanged)
2. Snapshot: copy density4→prevDensity4, velocity→prevVelocity (2 copies instead of 4)
3. Advect density4 (advectVelocity=0, useMacCormack=1)
   - Bind: prevDensity4→DENSITY4_INPUT, prevVelocity→VELOCITY_INPUT → density4→DENSITY4_OUTPUT
4. Barrier (density4 write → read)
5. Advect velocity (advectVelocity=1, useMacCormack=1)
   - Bind: prevVelocity→VELOCITY_INPUT, **density4→DENSITY4_INPUT** (read-only for buoyancy) → velocity→VELOCITY_OUTPUT
   - NOTE: density4 is bound as Texture3D (read-only) to DENSITY4_INPUT, NOT as RWTexture3D
6. Barrier
7. Splat — binds particle buffer → density4 (RW), velocity (RW)
8. Barrier
9. Pressure iterations (unchanged)
10. Project (unchanged)

**Intentional pass ordering**: buoyancy reads post-combustion, pre-splat density4. Freshly splatted fuel/temperature affects NEXT frame's combustion/buoyancy. This matches NvFlow and avoids single-frame feedback loops.

Update `rtx_context.cpp`:
- Replace 4× density + 4× temperature texture bindings with 4× density4 bindings (R16G16B16A16_SFLOAT)
- Replace blackbody LUT binding with colormap binding
- Update dummy texture format from R16_SFLOAT to R16G16B16A16_SFLOAT for unused density4 slots
- Update resolve args constant buffer (`particleVolumeArgs`) to use new field names
- Rename `ensureBlackbodyLUT()` → `ensureColormap()` (one-time lazy init, unchanged pattern)

Update `rtx_accel_manager.cpp`:
- Update `MemoryParticleVolume` field assignments to write `absorptionCrossSection`, `colorScale`, `alphaScale`, `shadowFactor` from `ActiveVolumeDescriptor`

Update `ActiveVolumeDescriptor` in `rtx_particle_system.h`:
- Replace `densityView`/`temperatureView` with `density4View`
- Replace `smokeDensity`/`emissionIntensityScale` with `absorptionCrossSection`, `colorScale`, `alphaScale`, `shadowFactor`

---

## Step 11: ImGui + RTX Options

**Files:**
- `src/dxvk/rtx_render/rtx_particle_system.h`
- `src/dxvk/rtx_render/rtx_particle_system.cpp`

**Actions:**

Replace current volume RTX_OPTIONs with NvFlow-style parameters:

```cpp
// Combustion
RTX_OPTION("rtx.particles.volume", float, volIgnitionTemp, 0.05f, "...");
RTX_OPTION("rtx.particles.volume", float, volBurnPerTemp, 4.0f, "...");
RTX_OPTION("rtx.particles.volume", float, volFuelPerBurn, 0.25f, "...");
RTX_OPTION("rtx.particles.volume", float, volTempPerBurn, 5.0f, "...");
RTX_OPTION("rtx.particles.volume", float, volSmokePerBurn, 3.0f, "...");
RTX_OPTION("rtx.particles.volume", float, volCoolingRate, 1.5f, "...");
RTX_OPTION("rtx.particles.volume", float, volDivergencePerBurn, 0.0f, "...");
RTX_OPTION("rtx.particles.volume", float, volEmitterCoupleRate, 3.0f, "...");
RTX_OPTION("rtx.particles.volume", float, volFuelAmount, 0.8f, "...");

// Damping (per-second values, converted to per-frame on CPU)
RTX_OPTION("rtx.particles.volume", float, volVelocityDamping, 0.01f, "...");
RTX_OPTION("rtx.particles.volume", float, volVelocityFade, 1.0f, "...");
RTX_OPTION("rtx.particles.volume", float, volSmokeDamping, 0.30f, "...");
RTX_OPTION("rtx.particles.volume", float, volSmokeFade, 0.65f, "...");

// Forces
RTX_OPTION("rtx.particles.volume", float, volBuoyancyPerTemp, 2.0f, "...");
RTX_OPTION("rtx.particles.volume", float, volBuoyancyPerSmoke, 0.0f, "...");
RTX_OPTION("rtx.particles.volume", float, volBuoyancyMaxSmoke, 1.0f, "...");
RTX_OPTION("rtx.particles.volume", float, volGravityMagnitude, 100.0f, "...");
RTX_OPTION("rtx.particles.volume", float, volVorticityConfinement, 0.6f, "...");
RTX_OPTION("rtx.particles.volume", float, volFluidCouplingStrength, 0.9f, "...");
RTX_OPTION("rtx.particles.volume", Vector3, volWindDirection, Vector3(0.f), "...");

// Rendering
RTX_OPTION("rtx.particles.volume", float, volAbsorptionCrossSection, 1.0f, "...");
RTX_OPTION("rtx.particles.volume", float, volColorScale, 1.0f, "...");
RTX_OPTION("rtx.particles.volume", float, volAlphaScale, 1.0f, "...");
RTX_OPTION("rtx.particles.volume", float, volShadowFactor, 0.5f, "...");

// Solver
RTX_OPTION("rtx.particles.volume", uint32_t, volPressureIterations, 30, "...");
```

**Constants setup** — compute damping rates on CPU:
```cpp
// NvFlow formula: dampingRate = 1 - pow(1 - damping, dt)
auto nfDamp = [](float damping, float dt) -> float {
  return 1.f - std::pow(1.f - std::min(damping, 1.f), dt);
};

volumeConstants.dampingRate = {
  0.f,                              // temp: no damping (cooling handles it)
  0.f,                              // fuel: no damping
  nfDamp(0.01f, dt),                // burn: light damping
  nfDamp(volSmokeDamping(), dt)     // smoke: moderate damping
};
volumeConstants.fadeRate = { 0.f, 0.f, 1.0f, volSmokeFade() };
volumeConstants.velDampingRate = vec4(nfDamp(volVelocityDamping(), dt));
volumeConstants.velFadeRate = vec4(volVelocityFade());
```

**ImGui** — organized into collapsible sections:
- Volume Fluid Simulation (master enable, memory budget, override toggle)
- Combustion (ignition, burn rates, cooling)
- Damping (velocity/smoke damping+fade sliders)
- Forces (buoyancy, gravity, wind, vorticity)
- Rendering (absorption, color/alpha scale, shadow factor)
- Solver (pressure iterations)

---

## Step 12: Test Scene

**Files:**
- `tests/rtx/apps/RemixAPI_C/remixapi_example_c.c`

**Actions:**

Update particle descriptor to use normalized values. The per-system descriptor fields that map to the new constants:
- `fuelAmount` = 0.8 (drives both temperature and fuel targets)
- `buoyancyCoefficient` → removed from per-system (now global `buoyancyPerTemp`)
- Other combustion params use global defaults

The test scene particle emitter should work without changes to spawn behavior — only the volume parameter values change.

---

## Step 13: Build + Smoke Test

**Actions:**
1. `meson compile -C _Comp64Release`
2. `meson install -C _Comp64Release`
3. Run `tests/rtx/dxvk_rt_testing/apics/RemixAPI_C/RemixAPI_C.exe`
4. Verify:
   - Volume allocates (check log)
   - Fire visible (colormap rendering)
   - Smoke visible (smoke channel + ambient scattering)
   - Smoke rises (buoyancy)
   - Simulation doesn't blow out after 30 seconds
   - ImGui controls respond
   - No GPU validation errors

---

## Parallelization Strategy

For implementation with subagents:

**Batch A (can be parallel):** Steps 3, 4, 5, 6, 7, 8, 9 — all shaders depend only on Step 1 (constants/layout) being done first.

**Sequential:** Step 1 → Step 2 → Batch A → Step 10 → Step 11 → Step 12 → Step 13

Realistic: Steps 1+2 first, then dispatch 4-5 subagents for the shader rewrites, then wire up C++ and test.
