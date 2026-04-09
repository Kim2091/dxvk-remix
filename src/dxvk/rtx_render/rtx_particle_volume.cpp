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

namespace dxvk {

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

} // namespace dxvk
