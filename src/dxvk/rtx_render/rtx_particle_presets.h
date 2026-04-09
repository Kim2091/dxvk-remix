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

#include <string>
#include "rtx_particle_system.h"

namespace dxvk {
namespace particlePresets {

// Small campfire: steady volumetric flame with gentle turbulence and slow dissipation.
inline RtxParticleSystemDesc campfire() {
  RtxParticleSystemDesc desc;
  desc.volumeType             = Volumetric;
  desc.smokeDensity           = 1.5f;
  desc.fuelAmount             = 0.8f;
  desc.burnTemperature        = 1200.0f;
  desc.buoyancyCoefficient    = 0.6f;
  desc.useTurbulence          = 1;
  desc.turbulenceFrequency    = 0.4f;
  desc.turbulenceForce        = 0.3f;
  desc.smokeDissipationRate   = 0.08f;
  desc.spawnRatePerSecond     = 20.0f;
  desc.maxNumParticles        = 200;
  desc.minTimeToLive          = 1.5f;
  desc.maxTimeToLive          = 3.0f;
  desc.initialVelocityFromNormal = 10.0f;
  desc.initialVelocityConeAngleDegrees = 30.0f;
  desc.gravityForce           = -5.0f;
  desc.dragCoefficient        = 0.2f;
  return desc;
}

// Explosion: hybrid burst with extreme heat, fast cooling, and strong buoyancy.
inline RtxParticleSystemDesc explosion() {
  RtxParticleSystemDesc desc;
  desc.volumeType             = Hybrid;
  desc.spawnBurstDuration     = 0.1f;
  desc.initialVelocityFromNormal = 200.0f;
  desc.burnTemperature        = 2500.0f;
  desc.coolingRate            = 0.6f;
  desc.buoyancyCoefficient    = 2.0f;
  desc.vorticityConfinement   = 0.8f;
  desc.smokeDensity           = 1.2f;
  desc.fuelAmount             = 0.9f;
  desc.smokeDissipationRate   = 0.2f;
  desc.spawnRatePerSecond     = 0.0f;
  desc.maxNumParticles        = 500;
  desc.minTimeToLive          = 0.5f;
  desc.maxTimeToLive          = 2.0f;
  desc.initialVelocityConeAngleDegrees = 60.0f;
  desc.gravityForce           = -2.0f;
  desc.dragCoefficient        = 0.1f;
  desc.useTurbulence          = 1;
  desc.turbulenceFrequency    = 0.8f;
  desc.turbulenceForce        = 1.0f;
  return desc;
}

// Chimney smoke: dense steady upward column with minimal vorticity and very slow dissipation.
inline RtxParticleSystemDesc chimneySmoke() {
  RtxParticleSystemDesc desc;
  desc.volumeType             = Volumetric;
  desc.smokeDensity           = 2.0f;
  desc.fuelAmount             = 0.0f;
  desc.initialVelocityFromNormal = 30.0f;
  desc.initialVelocityConeAngleDegrees = 10.0f;
  desc.vorticityConfinement   = 0.2f;
  desc.smokeDissipationRate   = 0.05f;
  desc.buoyancyCoefficient    = 0.8f;
  desc.spawnRatePerSecond     = 15.0f;
  desc.maxNumParticles        = 300;
  desc.minTimeToLive          = 4.0f;
  desc.maxTimeToLive          = 8.0f;
  desc.gravityForce           = -1.0f;
  desc.dragCoefficient        = 0.15f;
  desc.useTurbulence          = 1;
  desc.turbulenceFrequency    = 0.2f;
  desc.turbulenceForce        = 0.1f;
  return desc;
}

// Steam vent: low density, high-velocity burst with fast dissipation and low absorption.
inline RtxParticleSystemDesc steamVent() {
  RtxParticleSystemDesc desc;
  desc.volumeType             = Volumetric;
  desc.smokeDensity           = 0.4f;
  desc.fuelAmount             = 0.0f;
  desc.initialVelocityFromNormal = 80.0f;
  desc.initialVelocityConeAngleDegrees = 20.0f;
  desc.smokeDissipationRate   = 0.3f;
  desc.smokeAbsorptionCrossSection = 0.2f;
  desc.buoyancyCoefficient    = 0.5f;
  desc.spawnRatePerSecond     = 30.0f;
  desc.maxNumParticles        = 250;
  desc.minTimeToLive          = 0.5f;
  desc.maxTimeToLive          = 2.0f;
  desc.gravityForce           = -1.0f;
  desc.dragCoefficient        = 0.3f;
  desc.useTurbulence          = 1;
  desc.turbulenceFrequency    = 0.5f;
  desc.turbulenceForce        = 0.2f;
  return desc;
}

// Burning object: hybrid surface emission with sustained 1500K heat and moderate buoyancy.
inline RtxParticleSystemDesc burningObject() {
  RtxParticleSystemDesc desc;
  desc.volumeType             = Hybrid;
  desc.burnTemperature        = 1500.0f;
  desc.smokeDensity           = 1.8f;
  desc.fuelAmount             = 0.5f;
  desc.buoyancyCoefficient    = 0.8f;
  desc.coolingRate            = 0.3f;
  desc.smokeDissipationRate   = 0.1f;
  desc.vorticityConfinement   = 0.4f;
  desc.spawnRatePerSecond     = 25.0f;
  desc.maxNumParticles        = 350;
  desc.minTimeToLive          = 1.0f;
  desc.maxTimeToLive          = 3.5f;
  desc.initialVelocityFromNormal = 15.0f;
  desc.initialVelocityConeAngleDegrees = 45.0f;
  desc.gravityForce           = -3.0f;
  desc.dragCoefficient        = 0.2f;
  desc.useTurbulence          = 1;
  desc.turbulenceFrequency    = 0.3f;
  desc.turbulenceForce        = 0.4f;
  return desc;
}

// Smoldering: very dense low-fuel haze at dull heat with high vorticity and extremely slow dissipation.
inline RtxParticleSystemDesc smoldering() {
  RtxParticleSystemDesc desc;
  desc.volumeType             = Volumetric;
  desc.smokeDensity           = 2.5f;
  desc.fuelAmount             = 0.15f;
  desc.burnTemperature        = 700.0f;
  desc.buoyancyCoefficient    = 0.15f;
  desc.vorticityConfinement   = 0.6f;
  desc.smokeDissipationRate   = 0.04f;
  desc.coolingRate            = 0.1f;
  desc.spawnRatePerSecond     = 10.0f;
  desc.maxNumParticles        = 200;
  desc.minTimeToLive          = 5.0f;
  desc.maxTimeToLive          = 10.0f;
  desc.initialVelocityFromNormal = 5.0f;
  desc.initialVelocityConeAngleDegrees = 20.0f;
  desc.gravityForce           = -1.0f;
  desc.dragCoefficient        = 0.3f;
  desc.useTurbulence          = 1;
  desc.turbulenceFrequency    = 0.2f;
  desc.turbulenceForce        = 0.15f;
  return desc;
}

// Muzzle burst: hybrid instantaneous flash at 3500K with extreme velocity, strong drag, and fast volume decay.
inline RtxParticleSystemDesc muzzleBurst() {
  RtxParticleSystemDesc desc;
  desc.volumeType             = Hybrid;
  desc.spawnBurstDuration     = 0.05f;
  desc.initialVelocityFromNormal = 500.0f;
  desc.burnTemperature        = 3500.0f;
  desc.coolingRate            = 0.9f;
  desc.smokeDensity           = 0.8f;
  desc.dragCoefficient        = 0.8f;
  desc.volumeDecayTime        = 0.5f;
  desc.fuelAmount             = 0.7f;
  desc.buoyancyCoefficient    = 1.5f;
  desc.smokeDissipationRate   = 0.4f;
  desc.spawnRatePerSecond     = 0.0f;
  desc.maxNumParticles        = 150;
  desc.minTimeToLive          = 0.1f;
  desc.maxTimeToLive          = 0.5f;
  desc.initialVelocityConeAngleDegrees = 15.0f;
  desc.gravityForce           = -1.0f;
  desc.useTurbulence          = 0;
  return desc;
}

// Returns a pointer to a static preset instance by name, or nullptr if not found.
inline const RtxParticleSystemDesc* getPresetByName(const std::string& name) {
  static const RtxParticleSystemDesc s_campfire     = campfire();
  static const RtxParticleSystemDesc s_explosion    = explosion();
  static const RtxParticleSystemDesc s_chimneySmoke = chimneySmoke();
  static const RtxParticleSystemDesc s_steamVent    = steamVent();
  static const RtxParticleSystemDesc s_burningObject = burningObject();
  static const RtxParticleSystemDesc s_smoldering   = smoldering();
  static const RtxParticleSystemDesc s_muzzleBurst  = muzzleBurst();

  if (name == "campfire")      return &s_campfire;
  if (name == "explosion")     return &s_explosion;
  if (name == "chimneySmoke")  return &s_chimneySmoke;
  if (name == "steamVent")     return &s_steamVent;
  if (name == "burningObject") return &s_burningObject;
  if (name == "smoldering")    return &s_smoldering;
  if (name == "muzzleBurst")   return &s_muzzleBurst;

  return nullptr;
}

} // namespace particlePresets
} // namespace dxvk
