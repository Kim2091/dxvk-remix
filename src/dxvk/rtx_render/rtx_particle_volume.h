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
#pragma once

#include "../dxvk_format.h"
#include "../dxvk_include.h"

#include "rtx_resources.h"
#include "rtx_option.h"

namespace dxvk {

  class DxvkContext;

  // ParticleVolume manages a set of GPU 3D textures representing a volumetric
  // simulation cell (density, temperature, velocity, obstacles, pressure).
  // Resolution is chosen based on camera distance and may transition over time
  // with hysteresis to avoid thrashing.
  class ParticleVolume {
  public:
    // Grid resolution tiers, ordered coarsest to finest.
    enum class Resolution : uint32_t {
      Res16 = 16,
      Res32 = 32,
      Res64 = 64,
      Res128 = 128,
    };

    // RTX options controlling the particle volume system.
    RTX_OPTION("rtx.particles.volume", bool, enable, true,
               "Master enable for the particle volumetric smoke/fire system.");
    RTX_OPTION("rtx.particles.volume", uint32_t, memoryBudgetMB, 128,
               "Maximum GPU memory budget (in megabytes) for all active particle volumes.");
    RTX_OPTION("rtx.particles.volume", bool, accurateObstacles, false,
               "Enable AABB obstacle query for accurate obstacle representation inside the volume.");
    RTX_OPTION("rtx.particles.volume", float, nearDistanceThreshold, 5.0f,
               "Distance threshold in meters below which the volume uses 128^3 resolution.");
    RTX_OPTION("rtx.particles.volume", float, midDistanceThreshold, 20.0f,
               "Distance threshold in meters below which the volume uses 64^3 resolution.");
    RTX_OPTION("rtx.particles.volume", float, farDistanceThreshold, 50.0f,
               "Distance threshold in meters below which the volume uses 32^3 resolution. Above this threshold 16^3 is used.");
    RTX_OPTION("rtx.particles.volume", float, hysteresisTime, 0.5f,
               "Seconds that a pending resolution change must remain stable before the transition is applied.");

    ParticleVolume() = default;
    ~ParticleVolume();

    // Allocate (or reallocate) all GPU 3D textures at the requested resolution.
    void allocate(Rc<DxvkContext> ctx, Resolution resolution);

    // Release all GPU 3D texture resources.
    void release();

    // Returns true if GPU resources have been allocated.
    bool isAllocated() const;

    // Update the axis-aligned bounding box for this volume.
    // padding is added uniformly on all sides.
    // Returns the target resolution for the given camera distance.
    Resolution updateAABB(const Vector3& cameraPos,
                          const Vector3& minBounds,
                          const Vector3& maxBounds,
                          float padding);

    // Resample volume data into a new grid size (stub — no-op for now).
    void transitionResolution(Rc<DxvkContext> ctx, Resolution newResolution);

    // Returns the linear grid dimension (equal for all three axes).
    uint32_t gridDimension() const;

    // Returns the approximate GPU memory used by this volume in bytes.
    size_t memoryUsageBytes() const;

    // Accessor: current resolution.
    Resolution currentResolution() const { return m_resolution; }

    // GPU resource accessors — callers bind these as shader inputs/outputs.
    const Resources::Resource& densityTexture() const { return m_density; }
    const Resources::Resource& temperatureTexture() const { return m_temperature; }
    const Resources::Resource& velocityTexture() const { return m_velocity; }
    const Resources::Resource& prevVelocityTexture() const { return m_prevVelocity; }
    const Resources::Resource& obstacleTexture() const { return m_obstacle; }
    const Resources::Resource& pressureTexture(uint32_t pingPong) const { return m_pressure[pingPong & 1u]; }

    // Decay / live-particle tracking.
    bool hasLiveParticles() const { return m_hasLiveParticles; }
    void setHasLiveParticles(bool v) { m_hasLiveParticles = v; }
    float timeSinceLastParticle() const { return m_timeSinceLastParticle; }
    void setTimeSinceLastParticle(float t) { m_timeSinceLastParticle = t; }

  private:
    // Select resolution tier from camera distance.
    Resolution resolutionFromDistance(float distanceMeters) const;

    // --- GPU resources ---
    Resources::Resource m_density;        // R16_SFLOAT 3D — smoke/fire density
    Resources::Resource m_temperature;    // R16_SFLOAT 3D — temperature
    Resources::Resource m_velocity;       // R16G16B16A16_SFLOAT 3D — current frame velocity
    Resources::Resource m_prevVelocity;   // R16G16B16A16_SFLOAT 3D — previous frame velocity (two-way coupling)
    Resources::Resource m_obstacle;       // R8_UNORM 3D — obstacle mask
    Resources::Resource m_pressure[2];    // R16_SFLOAT 3D x2 — ping-pong pressure solve

    // --- Current state ---
    Resolution m_resolution = Resolution::Res32;
    Vector3 m_aabbMin = Vector3(0.f);
    Vector3 m_aabbMax = Vector3(0.f);

    // --- Hysteresis state ---
    Resolution m_pendingResolution = Resolution::Res32;
    float m_hysteresisTimer = 0.f;      // seconds elapsed since pending resolution was set

    // --- Decay tracking ---
    float m_timeSinceLastParticle = 0.f;
    bool m_hasLiveParticles = false;
  };

} // namespace dxvk
