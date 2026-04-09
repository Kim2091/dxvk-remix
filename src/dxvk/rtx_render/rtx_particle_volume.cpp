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
#include "rtx_particle_volume.h"

#include "dxvk_context.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"

#include "rtx/pass/particles/particle_volume_binding_indices.h"

#include <rtx_shaders/particle_volume_obstacle.h>
#include <rtx_shaders/particle_volume_splat.h>
#include <rtx_shaders/particle_volume_forces.h>
#include <rtx_shaders/particle_volume_pressure.h>
#include <rtx_shaders/particle_volume_project.h>
#include <rtx_shaders/particle_volume_advect.h>
#include <rtx_shaders/particle_volume_composite.h>

namespace dxvk {

  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {

    // ---------------------------------------------------------------------------
    // Obstacle pass: samples previous-frame GBuffer world position to mark solid
    // cells in the 3D obstacle texture.
    class ParticleVolumeObstacle : public ManagedShader {
      SHADER_SOURCE(ParticleVolumeObstacle, VK_SHADER_STAGE_COMPUTE_BIT, particle_volume_obstacle)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(PARTICLE_VOLUME_BINDING_CONSTANTS)
        TEXTURE2D(PARTICLE_VOLUME_BINDING_PREV_WORLD_POSITION_INPUT)
        RW_TEXTURE3D(PARTICLE_VOLUME_BINDING_OBSTACLE_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ParticleVolumeObstacle);

    // ---------------------------------------------------------------------------
    // Splat pass: scatters per-particle density and temperature into the 3D grid
    // using trilinear weighting.
    class ParticleVolumeSplat : public ManagedShader {
      SHADER_SOURCE(ParticleVolumeSplat, VK_SHADER_STAGE_COMPUTE_BIT, particle_volume_splat)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(PARTICLE_VOLUME_BINDING_CONSTANTS)
        STRUCTURED_BUFFER(PARTICLE_VOLUME_BINDING_PARTICLES_INPUT)
        RW_TEXTURE3D(PARTICLE_VOLUME_BINDING_DENSITY_OUTPUT)
        RW_TEXTURE3D(PARTICLE_VOLUME_BINDING_TEMPERATURE_OUTPUT)
        RW_TEXTURE3D(PARTICLE_VOLUME_BINDING_VELOCITY_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ParticleVolumeSplat);

    // ---------------------------------------------------------------------------
    // Forces pass: applies buoyancy, wind, and vorticity confinement to the
    // velocity field.  Inputs use HLSL register(t, space1); outputs use space2.
    class ParticleVolumeForces : public ManagedShader {
      SHADER_SOURCE(ParticleVolumeForces, VK_SHADER_STAGE_COMPUTE_BIT, particle_volume_forces)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(PARTICLE_VOLUME_BINDING_CONSTANTS)
        TEXTURE3D(PARTICLE_VOLUME_BINDING_TEMPERATURE_INPUT)
        TEXTURE3D(PARTICLE_VOLUME_BINDING_OBSTACLE_INPUT)
        RW_TEXTURE3D(PARTICLE_VOLUME_BINDING_VELOCITY_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ParticleVolumeForces);

    // ---------------------------------------------------------------------------
    // Pressure pass: one Jacobi iteration step of the pressure Poisson solve.
    // Ping-ponged N times between the two pressure textures.
    class ParticleVolumePressure : public ManagedShader {
      SHADER_SOURCE(ParticleVolumePressure, VK_SHADER_STAGE_COMPUTE_BIT, particle_volume_pressure)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(PARTICLE_VOLUME_BINDING_CONSTANTS)
        TEXTURE3D(PARTICLE_VOLUME_BINDING_VELOCITY_INPUT)
        TEXTURE3D(PARTICLE_VOLUME_BINDING_PRESSURE_INPUT)
        TEXTURE3D(PARTICLE_VOLUME_BINDING_OBSTACLE_INPUT)
        RW_TEXTURE3D(PARTICLE_VOLUME_BINDING_PRESSURE_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ParticleVolumePressure);

    // ---------------------------------------------------------------------------
    // Project pass: subtracts the pressure gradient from the velocity field to
    // enforce incompressibility.
    class ParticleVolumeProject : public ManagedShader {
      SHADER_SOURCE(ParticleVolumeProject, VK_SHADER_STAGE_COMPUTE_BIT, particle_volume_project)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(PARTICLE_VOLUME_BINDING_CONSTANTS)
        TEXTURE3D(PARTICLE_VOLUME_BINDING_PRESSURE_INPUT)
        TEXTURE3D(PARTICLE_VOLUME_BINDING_OBSTACLE_INPUT)
        RW_TEXTURE3D(PARTICLE_VOLUME_BINDING_VELOCITY_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ParticleVolumeProject);

    // ---------------------------------------------------------------------------
    // Advect pass: semi-Lagrangian advection.
    // Dispatched twice: once for density/temperature (advectVelocity=0), once for
    // the velocity field itself (advectVelocity=1).
    struct AdvectPushConstants {
      uint32_t advectVelocity; // 0 = advect density/temp, 1 = advect velocity
      uint32_t useMacCormack;  // reserved: 0 = basic semi-Lagrangian
    };

    class ParticleVolumeAdvect : public ManagedShader {
      SHADER_SOURCE(ParticleVolumeAdvect, VK_SHADER_STAGE_COMPUTE_BIT, particle_volume_advect)

      PUSH_CONSTANTS(AdvectPushConstants)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(PARTICLE_VOLUME_BINDING_CONSTANTS)
        TEXTURE3D(PARTICLE_VOLUME_BINDING_VELOCITY_INPUT)
        TEXTURE3D(PARTICLE_VOLUME_BINDING_DENSITY_INPUT)
        TEXTURE3D(PARTICLE_VOLUME_BINDING_TEMPERATURE_INPUT)
        TEXTURE3D(PARTICLE_VOLUME_BINDING_OBSTACLE_INPUT)
        SAMPLER(PARTICLE_VOLUME_BINDING_LINEAR_SAMPLER)
        RW_TEXTURE3D(PARTICLE_VOLUME_BINDING_DENSITY_OUTPUT)
        RW_TEXTURE3D(PARTICLE_VOLUME_BINDING_TEMPERATURE_OUTPUT)
        RW_TEXTURE3D(PARTICLE_VOLUME_BINDING_VELOCITY_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ParticleVolumeAdvect);

    // ---------------------------------------------------------------------------
    // Composite pass: screen-space ray-march that blends volume fire/smoke into
    // the composited color buffer.
    class ParticleVolumeComposite : public ManagedShader {
      SHADER_SOURCE(ParticleVolumeComposite, VK_SHADER_STAGE_COMPUTE_BIT, particle_volume_composite)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(PARTICLE_VOLUME_BINDING_CONSTANTS)
        TEXTURE3D(PARTICLE_VOLUME_BINDING_DENSITY_INPUT)
        TEXTURE3D(PARTICLE_VOLUME_BINDING_TEMPERATURE_INPUT)
        SAMPLER2D(PARTICLE_VOLUME_BINDING_BLACKBODY_LUT_INPUT)
        SAMPLER(PARTICLE_VOLUME_BINDING_LINEAR_SAMPLER)
        TEXTURE2D(PARTICLE_VOLUME_BINDING_COMPOSITE_WORLD_POS_INPUT)
        RW_TEXTURE2D(PARTICLE_VOLUME_BINDING_COMPOSITE_COLOR_INOUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ParticleVolumeComposite);

  } // anonymous namespace


  // Bytes per grid cell across all textures:
  //   density     : R16_SFLOAT          = 2 bytes
  //   temperature : R16_SFLOAT          = 2 bytes
  //   velocity    : R16G16B16A16_SFLOAT = 8 bytes
  //   prevVelocity: R16G16B16A16_SFLOAT = 8 bytes
  //   obstacle    : R8_UNORM            = 1 byte
  //   pressure[0] : R16_SFLOAT          = 2 bytes
  //   pressure[1] : R16_SFLOAT          = 2 bytes
  //   total                             = 25 bytes
  static constexpr size_t kBytesPerCell = 25u;

  // ---------------------------------------------------------------------------
  ParticleVolume::~ParticleVolume() {
    release();
  }

  // ---------------------------------------------------------------------------
  void ParticleVolume::allocate(Rc<DxvkContext>& ctx, Resolution resolution) {
    ScopedCpuProfileZone();

    // Release any existing resources before reallocating.
    release();

    m_resolution = resolution;

    const uint32_t dim = gridDimension();
    const VkExtent3D extent = { dim, dim, dim };

    m_density = Resources::createImageResource(ctx, "particle volume density",
      extent, VK_FORMAT_R16_SFLOAT, 1, VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D);

    m_temperature = Resources::createImageResource(ctx, "particle volume temperature",
      extent, VK_FORMAT_R16_SFLOAT, 1, VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D);

    m_velocity = Resources::createImageResource(ctx, "particle volume velocity",
      extent, VK_FORMAT_R16G16B16A16_SFLOAT, 1, VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D);

    m_prevVelocity = Resources::createImageResource(ctx, "particle volume prev velocity",
      extent, VK_FORMAT_R16G16B16A16_SFLOAT, 1, VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D);

    m_obstacle = Resources::createImageResource(ctx, "particle volume obstacle",
      extent, VK_FORMAT_R8_UNORM, 1, VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D);

    m_pressure[0] = Resources::createImageResource(ctx, "particle volume pressure 0",
      extent, VK_FORMAT_R16_SFLOAT, 1, VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D);

    m_pressure[1] = Resources::createImageResource(ctx, "particle volume pressure 1",
      extent, VK_FORMAT_R16_SFLOAT, 1, VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D);

    // Constant buffer for per-frame volume simulation parameters.
    {
      DxvkBufferCreateInfo info;
      info.size   = sizeof(ParticleVolumeConstants);
      info.usage  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
      m_cb = ctx->getDevice()->createBuffer(
        info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXBuffer,
        "RTX Particle Volume - Constant Buffer");
    }

    // Reset hysteresis so the new resolution is treated as stable.
    m_pendingResolution = resolution;
    m_hysteresisTimer = 0.f;
  }

  // ---------------------------------------------------------------------------
  void ParticleVolume::release() {
    if (!isAllocated()) return;
    ScopedCpuProfileZone();

    m_density.reset();
    m_temperature.reset();
    m_velocity.reset();
    m_prevVelocity.reset();
    m_obstacle.reset();
    m_pressure[0].reset();
    m_pressure[1].reset();
    m_cb = nullptr;
  }

  // ---------------------------------------------------------------------------
  bool ParticleVolume::isAllocated() const {
    return m_density.isValid();
  }

  // ---------------------------------------------------------------------------
  ParticleVolume::Resolution ParticleVolume::resolutionFromDistance(float distanceMeters) const {
    if (distanceMeters < nearDistanceThreshold()) {
      return Resolution::Res128;
    } else if (distanceMeters < midDistanceThreshold()) {
      return Resolution::Res64;
    } else if (distanceMeters < farDistanceThreshold()) {
      return Resolution::Res32;
    } else {
      return Resolution::Res16;
    }
  }

  // ---------------------------------------------------------------------------
  ParticleVolume::Resolution ParticleVolume::updateAABB(const Vector3& cameraPos,
                                                        const Vector3& minBounds,
                                                        const Vector3& maxBounds,
                                                        float padding) {
    ScopedCpuProfileZone();

    // Apply padding uniformly on all sides.
    m_aabbMin = minBounds - Vector3(padding);
    m_aabbMax = maxBounds + Vector3(padding);

    // Compute the closest point on the AABB to the camera and measure distance.
    const Vector3 clamped(
      std::max(m_aabbMin.x, std::min(cameraPos.x, m_aabbMax.x)),
      std::max(m_aabbMin.y, std::min(cameraPos.y, m_aabbMax.y)),
      std::max(m_aabbMin.z, std::min(cameraPos.z, m_aabbMax.z))
    );
    const Vector3 delta = cameraPos - clamped;
    const float distanceMeters = std::sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);

    return resolutionFromDistance(distanceMeters);
  }

  // ---------------------------------------------------------------------------
  void ParticleVolume::transitionResolution(Rc<DxvkContext>& ctx, Resolution newResolution) {
    ScopedCpuProfileZone();

    // Stub: reallocate at the new resolution without resampling existing data.
    // A future implementation can trilinearly resample m_density / m_temperature
    // into a temporary buffer and blit into the new textures.
    allocate(ctx, newResolution);
  }

  // ---------------------------------------------------------------------------
  uint32_t ParticleVolume::gridDimension() const {
    return static_cast<uint32_t>(m_resolution);
  }

  // ---------------------------------------------------------------------------
  size_t ParticleVolume::memoryUsageBytes() const {
    if (!isAllocated()) {
      return 0u;
    }
    const uint32_t dim = gridDimension();
    const size_t cellCount = static_cast<size_t>(dim) * dim * dim;
    return cellCount * kBytesPerCell;
  }

  // ---------------------------------------------------------------------------
  void ParticleVolume::simulateFluid(
      Rc<DxvkContext>& ctx,
      const ParticleVolumeConstants& constants,
      Rc<DxvkBuffer> pParticleBuffer,
      uint32_t particleCount,
      Rc<DxvkImageView> prevWorldPosView) {
    ScopedCpuProfileZone();

    if (!isAllocated()) {
      return;
    }

    // Upload the per-frame constants once; every pass shares the same CB.
    ctx->writeToBuffer(m_cb, 0, sizeof(ParticleVolumeConstants), &constants);
    ctx->bindResourceBuffer(PARTICLE_VOLUME_BINDING_CONSTANTS, DxvkBufferSlice(m_cb));

    const uint32_t dim = gridDimension();
    const uint32_t groups3D = (dim + 7u) / 8u;

    // Helper: emit a compute-to-compute memory barrier.
    auto emitComputeBarrier = [&]() {
      ctx->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
    };

    // -------------------------------------------------------------------------
    // Pass 1 — Obstacle
    // (skipped when prevWorldPosView is null — obstacle texture remains as-is)
    // -------------------------------------------------------------------------
    if (prevWorldPosView != nullptr) {
      ScopedGpuProfileZone(ctx, "ParticleVolume_Obstacle");

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ParticleVolumeObstacle::getShader());

      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_PREV_WORLD_POSITION_INPUT,
        prevWorldPosView, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_OBSTACLE_OUTPUT,
        m_obstacle.view, nullptr);

      ctx->dispatch(groups3D, groups3D, groups3D);

      emitComputeBarrier();
    }

    // -------------------------------------------------------------------------
    // Pass 2 — Splat (one thread per particle)
    // -------------------------------------------------------------------------
    {
      ScopedGpuProfileZone(ctx, "ParticleVolume_Splat");

      if (particleCount > 0 && pParticleBuffer != nullptr) {
        ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ParticleVolumeSplat::getShader());

        ctx->bindResourceBuffer(PARTICLE_VOLUME_BINDING_PARTICLES_INPUT,
          DxvkBufferSlice(pParticleBuffer));
        ctx->bindResourceView(PARTICLE_VOLUME_BINDING_DENSITY_OUTPUT,
          m_density.view, nullptr);
        ctx->bindResourceView(PARTICLE_VOLUME_BINDING_TEMPERATURE_OUTPUT,
          m_temperature.view, nullptr);
        ctx->bindResourceView(PARTICLE_VOLUME_BINDING_VELOCITY_OUTPUT,
          m_velocity.view, nullptr);

        const uint32_t splatGroups = (particleCount + 127u) / 128u;
        ctx->dispatch(splatGroups, 1, 1);
      }
    }

    emitComputeBarrier();

    // -------------------------------------------------------------------------
    // Pass 3 — Forces (buoyancy, wind, vorticity)
    // -------------------------------------------------------------------------
    {
      ScopedGpuProfileZone(ctx, "ParticleVolume_Forces");

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ParticleVolumeForces::getShader());

      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_TEMPERATURE_INPUT,
        m_temperature.view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_OBSTACLE_INPUT,
        m_obstacle.view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_VELOCITY_OUTPUT,
        m_velocity.view, nullptr);

      ctx->dispatch(groups3D, groups3D, groups3D);
    }

    emitComputeBarrier();

    // -------------------------------------------------------------------------
    // Pass 4 — Pressure Jacobi iterations (ping-pong)
    // -------------------------------------------------------------------------
    {
      ScopedGpuProfileZone(ctx, "ParticleVolume_Pressure");

      const uint32_t iterations = constants.pressureIterations > 0
        ? constants.pressureIterations
        : 20u;

      for (uint32_t i = 0u; i < iterations; ++i) {
        const uint32_t src = i & 1u;
        const uint32_t dst = 1u - src;

        ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ParticleVolumePressure::getShader());

        ctx->bindResourceView(PARTICLE_VOLUME_BINDING_VELOCITY_INPUT,
          m_velocity.view, nullptr);
        ctx->bindResourceView(PARTICLE_VOLUME_BINDING_PRESSURE_INPUT,
          m_pressure[src].view, nullptr);
        ctx->bindResourceView(PARTICLE_VOLUME_BINDING_OBSTACLE_INPUT,
          m_obstacle.view, nullptr);
        ctx->bindResourceView(PARTICLE_VOLUME_BINDING_PRESSURE_OUTPUT,
          m_pressure[dst].view, nullptr);

        ctx->dispatch(groups3D, groups3D, groups3D);

        emitComputeBarrier();
      }
    }

    // -------------------------------------------------------------------------
    // Pass 4b — Project (subtract pressure gradient, enforce incompressibility)
    // -------------------------------------------------------------------------
    {
      ScopedGpuProfileZone(ctx, "ParticleVolume_Project");

      // After N iterations the latest pressure result is in pressure[(N-1)&1 ^ 1]
      // = pressure[iterations & 1].  Use that as the final pressure input.
      const uint32_t iterations = constants.pressureIterations > 0
        ? constants.pressureIterations
        : 20u;
      const uint32_t finalPressureSrc = iterations & 1u;

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ParticleVolumeProject::getShader());

      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_PRESSURE_INPUT,
        m_pressure[finalPressureSrc].view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_OBSTACLE_INPUT,
        m_obstacle.view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_VELOCITY_OUTPUT,
        m_velocity.view, nullptr);

      ctx->dispatch(groups3D, groups3D, groups3D);
    }

    emitComputeBarrier();

    // Copy projected velocity -> prevVelocity for two-way coupling next frame.
    {
      VkImageSubresourceLayers subresource;
      subresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      subresource.mipLevel       = 0;
      subresource.baseArrayLayer = 0;
      subresource.layerCount     = 1;

      ctx->copyImage(
        m_prevVelocity.image,
        subresource,
        VkOffset3D { 0, 0, 0 },
        m_velocity.image,
        subresource,
        VkOffset3D { 0, 0, 0 },
        VkExtent3D { dim, dim, dim });
    }

    emitComputeBarrier();

    // Create a linear-clamp sampler for the advection trilinear sampling.
    Rc<DxvkSampler> linearSampler;
    {
      DxvkSamplerCreateInfo samplerInfo {};
      samplerInfo.magFilter      = VK_FILTER_LINEAR;
      samplerInfo.minFilter      = VK_FILTER_LINEAR;
      samplerInfo.mipmapMode     = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      samplerInfo.mipmapLodBias  = 0.f;
      samplerInfo.mipmapLodMin   = 0.f;
      samplerInfo.mipmapLodMax   = 0.f;
      samplerInfo.useAnisotropy  = VK_FALSE;
      samplerInfo.maxAnisotropy  = 1.f;
      samplerInfo.addressModeU   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      samplerInfo.addressModeV   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      samplerInfo.addressModeW   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      samplerInfo.compareToDepth = VK_FALSE;
      samplerInfo.compareOp      = VK_COMPARE_OP_ALWAYS;
      samplerInfo.borderColor    = {};
      samplerInfo.usePixelCoord  = VK_FALSE;
      linearSampler = ctx->getDevice()->createSampler(samplerInfo);
    }

    // -------------------------------------------------------------------------
    // Pass 5 — Advect density and temperature (advectVelocity = 0)
    // -------------------------------------------------------------------------
    {
      ScopedGpuProfileZone(ctx, "ParticleVolume_AdvectDensityTemp");

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ParticleVolumeAdvect::getShader());

      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_VELOCITY_INPUT,
        m_velocity.view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_DENSITY_INPUT,
        m_density.view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_TEMPERATURE_INPUT,
        m_temperature.view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_OBSTACLE_INPUT,
        m_obstacle.view, nullptr);
      ctx->bindResourceSampler(PARTICLE_VOLUME_BINDING_LINEAR_SAMPLER, linearSampler);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_DENSITY_OUTPUT,
        m_density.view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_TEMPERATURE_OUTPUT,
        m_temperature.view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_VELOCITY_OUTPUT,
        m_velocity.view, nullptr);

      ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
      const AdvectPushConstants pushConst { 0u, 0u };
      ctx->pushConstants(0, sizeof(pushConst), &pushConst);

      ctx->dispatch(groups3D, groups3D, groups3D);
    }

    emitComputeBarrier();

    // -------------------------------------------------------------------------
    // Pass 6 — Advect velocity (advectVelocity = 1)
    // -------------------------------------------------------------------------
    {
      ScopedGpuProfileZone(ctx, "ParticleVolume_AdvectVelocity");

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ParticleVolumeAdvect::getShader());

      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_VELOCITY_INPUT,
        m_velocity.view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_DENSITY_INPUT,
        m_density.view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_TEMPERATURE_INPUT,
        m_temperature.view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_OBSTACLE_INPUT,
        m_obstacle.view, nullptr);
      ctx->bindResourceSampler(PARTICLE_VOLUME_BINDING_LINEAR_SAMPLER, linearSampler);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_DENSITY_OUTPUT,
        m_density.view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_TEMPERATURE_OUTPUT,
        m_temperature.view, nullptr);
      ctx->bindResourceView(PARTICLE_VOLUME_BINDING_VELOCITY_OUTPUT,
        m_velocity.view, nullptr);

      ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
      const AdvectPushConstants pushConst { 1u, 0u };
      ctx->pushConstants(0, sizeof(pushConst), &pushConst);

      ctx->dispatch(groups3D, groups3D, groups3D);
    }
  }

  // ---------------------------------------------------------------------------
  void ParticleVolume::compositeVolume(
      Rc<DxvkContext>& ctx,
      const ParticleVolumeConstants& constants,
      Rc<DxvkImageView> worldPosView,
      Rc<DxvkImageView> colorView,
      Rc<DxvkImageView> blackbodyLUTView) {
    ScopedCpuProfileZone();

    if (!isAllocated()) {
      return;
    }

    ScopedGpuProfileZone(ctx, "ParticleVolume_Composite");

    // Upload the per-frame constants.
    ctx->writeToBuffer(m_cb, 0, sizeof(ParticleVolumeConstants), &constants);
    ctx->bindResourceBuffer(PARTICLE_VOLUME_BINDING_CONSTANTS, DxvkBufferSlice(m_cb));

    // Create a linear-clamp sampler for volume sampling and blackbody LUT.
    Rc<DxvkSampler> linearSampler;
    {
      DxvkSamplerCreateInfo samplerInfo {};
      samplerInfo.magFilter      = VK_FILTER_LINEAR;
      samplerInfo.minFilter      = VK_FILTER_LINEAR;
      samplerInfo.mipmapMode     = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      samplerInfo.mipmapLodBias  = 0.f;
      samplerInfo.mipmapLodMin   = 0.f;
      samplerInfo.mipmapLodMax   = 0.f;
      samplerInfo.useAnisotropy  = VK_FALSE;
      samplerInfo.maxAnisotropy  = 1.f;
      samplerInfo.addressModeU   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      samplerInfo.addressModeV   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      samplerInfo.addressModeW   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      samplerInfo.compareToDepth = VK_FALSE;
      samplerInfo.compareOp      = VK_COMPARE_OP_ALWAYS;
      samplerInfo.borderColor    = {};
      samplerInfo.usePixelCoord  = VK_FALSE;
      linearSampler = ctx->getDevice()->createSampler(samplerInfo);
    }

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ParticleVolumeComposite::getShader());

    // Bind volume textures (read-only for composite).
    ctx->bindResourceView(PARTICLE_VOLUME_BINDING_DENSITY_INPUT,
      m_density.view, nullptr);
    ctx->bindResourceView(PARTICLE_VOLUME_BINDING_TEMPERATURE_INPUT,
      m_temperature.view, nullptr);

    // Bind blackbody LUT.
    ctx->bindResourceView(PARTICLE_VOLUME_BINDING_BLACKBODY_LUT_INPUT,
      blackbodyLUTView, nullptr);
    ctx->bindResourceSampler(PARTICLE_VOLUME_BINDING_BLACKBODY_LUT_INPUT,
      linearSampler);

    // Bind the linear sampler for volume trilinear sampling.
    ctx->bindResourceSampler(PARTICLE_VOLUME_BINDING_LINEAR_SAMPLER, linearSampler);

    // Bind GBuffer world position (current frame).
    ctx->bindResourceView(PARTICLE_VOLUME_BINDING_COMPOSITE_WORLD_POS_INPUT,
      worldPosView, nullptr);

    // Bind the composited color buffer (read-write).
    ctx->bindResourceView(PARTICLE_VOLUME_BINDING_COMPOSITE_COLOR_INOUT,
      colorView, nullptr);

    // Dispatch: one thread per pixel at composite resolution.
    const uint32_t groupsX = (constants.renderingWidth  + 15u) / 16u;
    const uint32_t groupsY = (constants.renderingHeight +  7u) /  8u;
    ctx->dispatch(groupsX, groupsY, 1);
  }

} // namespace dxvk
