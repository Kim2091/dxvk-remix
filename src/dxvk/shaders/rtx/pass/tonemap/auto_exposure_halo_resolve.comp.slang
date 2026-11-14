/*
* Halo Meter auto exposure, pass 2 of 2: fold the tiles, turn them into an exposure and
* move the stored one towards it.
*
* Provenance: a Slang port of the `MeterResolve` compute entry point in
* shaders/composite.hlsl of the "halocedlaart" Halo CE (MCC) path-tracer ReShade addon
* (archive halocedlaart5.rar). See auto_exposure_halo_meter.comp.slang for the
* attribution note.
*
* The wanted exposure is clamp(target / geometricMean * compensation, 2^minEv, 2^maxEv).
* The blend is linear in the exposure multiplier, with the fraction computed on the CPU.
* The first reading after a reset snaps instead of fading up.
* The result goes to the shared 1x1 exposure texture that all downstream consumers read.
*/
#include "rtx/pass/tonemap/tonemapping.h"

// x = sum(w * log), y = sum(w), z = in-scene surface centre weight, w = total centre weight.
layout(rgba32f, binding = AUTO_EXPOSURE_HALO_PARTIALS_INPUT_OUTPUT)
RWTexture2D<float4> InPartials;

// x = adapted exposure, y = 1 once anything has been metered since the last reset, z = the running average of the
// excluded view model share, w = the seconds the current view model spike has held the exposure.
layout(rgba32f, binding = AUTO_EXPOSURE_HALO_STATE_INPUT_OUTPUT)
RWTexture1D<float4> InOutState;

layout(r32f, binding = AUTO_EXPOSURE_HALO_EXPOSURE_OUTPUT)
RWTexture1D<float> OutExposure;

layout(push_constant)
ConstantBuffer<ToneMappingHaloMeterArgs> cb;

static const float kMeterFloor = 1e-4f;
// The time constant of the running average of the excluded view model share, in seconds.
static const float kViewModelShareTimeConstant = 1.0f;
// The longest a view model spike holds the exposure, in seconds, so a lasting change such as a bigger weapon adapts.
static const float kViewModelMaxHoldSeconds = 0.5f;
static const uint kResolveThreads = 256u;

groupshared float g_meterLogSum[kResolveThreads];
groupshared float g_meterWeightSum[kResolveThreads];
groupshared float g_meterSurfaceSum[kResolveThreads];
groupshared float g_meterCentreSum[kResolveThreads];
// The metered weight with the view model included, for the share the exclusion removed.
groupshared float g_meterAllWeightSum[kResolveThreads];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint index : SV_GroupIndex)
{
  const uint groups = cb.groupsX * cb.groupsY;

  // Each tile has the sums without the excluded view model pixels, then its metered weight with them.
  float4 sum = float4(0.0f);
  float allWeight = 0.0f;
  for (uint i = index; i < groups; i += kResolveThreads)
  {
    const uint2 tile = uint2((i % cb.groupsX) * 2u, i / cb.groupsX);
    sum += InPartials[tile];
    allWeight += InPartials[tile + uint2(1u, 0u)].y;
  }

  g_meterLogSum[index] = sum.x;
  g_meterWeightSum[index] = sum.y;
  g_meterSurfaceSum[index] = sum.z;
  g_meterCentreSum[index] = sum.w;
  g_meterAllWeightSum[index] = allWeight;

  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = kResolveThreads / 2u; stride > 0u; stride >>= 1u)
  {
    if (index < stride)
    {
      g_meterLogSum[index] += g_meterLogSum[index + stride];
      g_meterWeightSum[index] += g_meterWeightSum[index + stride];
      g_meterSurfaceSum[index] += g_meterSurfaceSum[index + stride];
      g_meterCentreSum[index] += g_meterCentreSum[index + stride];
      g_meterAllWeightSum[index] += g_meterAllWeightSum[index + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (index != 0u)
  {
    return;
  }

  // A non-finite or non-positive stored exposure would never recover through the blend,
  // so it counts as unmetered and the next reading snaps.
  const float4 state = InOutState[0];
  float2 previous = state.xy;
  if (!isfinite(previous.x) || !(previous.x > 0.0f))
  {
    previous = float2(1.0f, 0.0f);
  }

  // Hold when there is no weight, or when scene surface covers less than minCoverage
  // of the centre-weighted frame.
  const float coverage = g_meterCentreSum[0] > 0.0f ? g_meterSurfaceSum[0] / g_meterCentreSum[0] : 0.0f;
  // Hold too while the excluded view model share spikes above its running average by more than the spike margin, as
  // when a sight kicks back into the frame, or exceeds the hold threshold, so the few pixels left cannot swing the
  // exposure. A spike holds for at most kViewModelMaxHoldSeconds. The average follows the share only while it does not
  // hold, and starts from the current share after a reset or camera cut.
  const float excludedShare = g_meterAllWeightSum[0] > 1e-6f ? 1.0f - g_meterWeightSum[0] / g_meterAllWeightSum[0] : 0.0f;
  const bool stateValid = previous.y > 0.5f && isfinite(state.z) && isfinite(state.w);
  float averageShare = stateValid ? state.z : excludedShare;
  float heldSeconds = stateValid ? max(state.w, 0.0f) : 0.0f;
  const bool viewModelSpike = cb.excludeViewModel != 0 &&
    (excludedShare > averageShare + cb.viewModelSpikeMargin || excludedShare > cb.viewModelHoldThreshold);
  const bool viewModelHold = viewModelSpike && heldSeconds < kViewModelMaxHoldSeconds;

  if (viewModelHold)
  {
    heldSeconds += cb.frameSeconds;
  }
  else
  {
    // After a spike outlasts the hold, the held time stays until the spike ends, so it does not hold again.
    heldSeconds = viewModelSpike ? heldSeconds : 0.0f;
    averageShare = lerp(averageShare, excludedShare, 1.0f - exp(-cb.frameSeconds / kViewModelShareTimeConstant));
  }

  if (g_meterWeightSum[0] <= 1e-6f || coverage < cb.minCoverage || viewModelHold)
  {
    InOutState[0] = float4(previous, averageShare, heldSeconds);
    if (previous.y > 0.5f)
    {
      OutExposure[0] = previous.x;  // Keep the adapted exposure.
    }
    else
    {
      // Nothing has been metered since the reset, so stay neutral and unmetered.
      OutExposure[0] = 1.0f;
    }
    return;
  }

  // The mean floor stops a near-black frame from running the exposure away.
  const float mean = max(exp(g_meterLogSum[0] / max(g_meterWeightSum[0], 1e-6f)), cb.meanFloor);
  if (!isfinite(mean))
  {
    // Nothing valid was metered this frame, so hold.
    OutExposure[0] = previous.y > 0.5f ? previous.x : 1.0f;
    InOutState[0] = float4(previous, averageShare, heldSeconds);
    return;
  }
  // The scene's EV100 sets a compensation bias, so night can sit a few stops under mid grey.
  const float sceneEv = log2(max(mean, kMeterFloor)) + cb.evOffset;
  const float nightWeight = cb.compensationDayEv > cb.compensationNightEv
    ? saturate((cb.compensationDayEv - sceneEv) / (cb.compensationDayEv - cb.compensationNightEv))
    : (sceneEv <= cb.compensationNightEv ? 1.0f : 0.0f);
  const float compensation = exp2(cb.compensationNightStops * nightWeight);
  const float wanted = clamp(cb.target / max(mean, kMeterFloor) * compensation,
                             exp2(cb.minEv), exp2(cb.maxEv));

  // State .y is 1 once anything has been metered since the last reset. Until then the blend snaps.
  const float blend = previous.y > 0.5f ? saturate(cb.adaptation) : 1.0f;
  const float exposure = lerp(previous.x, wanted, blend);

  InOutState[0] = float4(exposure, 1.0f, averageShare, heldSeconds);
  OutExposure[0] = exposure;
}
