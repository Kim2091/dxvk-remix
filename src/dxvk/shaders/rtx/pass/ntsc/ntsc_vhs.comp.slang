/*
 * NTSC/VHS tape-path post-process.
 *
 * The source texture is post-tonemap linear RGB, while the tape model works
 * in gamma-coded RGB inside a composite signal. This pass applies the
 * equivalent tape path directly in sRGB/YIQ space:
 *
 *   pass 0: luma/color-under bandwidth -> luma edge ringing
 *   pass 1: head-wear smear -> band-limited luma noise
 *   pass 2: tape dropout
 *   pass 3: causal luma-only tape trail -> linear RGB
 */
#include "rtx/pass/ntsc/ntsc_vhs.h"

layout(binding = NTSC_VHS_INPUT)
Sampler2D InColorSampler;

layout(binding = NTSC_VHS_OUTPUT)
RWTexture2D<float4> OutColorTexture;

layout(push_constant)
ConstantBuffer<NtscVhsArgs> cb;

static const float PI = 3.14159265f;
static const float TAU = 6.28318531f;
static const float3 LUMA = float3(0.299f, 0.587f, 0.114f);
static const float ACTIVE_SAMPLES = 754.0f;
static const float SAMPLE_RATE = 14318180.24f;
static const float NYQUIST = SAMPLE_RATE * 0.5f;

float hash(float2 v) {
  return frac(sin(dot(v, float2(12.9898f, 78.233f))) * 43758.5453f);
}

float hash3(float3 v) {
  return frac(sin(dot(v, float3(12.9898f, 78.233f, 37.719f))) * 43758.5453f);
}

float3 toYIQ(float3 c) {
  return float3(
    dot(c, LUMA),
    dot(c, float3(0.595901f, -0.274557f, -0.321344f)),
    dot(c, float3(0.211537f, -0.522736f, 0.311200f)));
}

float3 toRGB(float3 v) {
  return float3(
    dot(v, float3(1.0f, 0.956f, 0.621f)),
    dot(v, float3(1.0f, -0.272f, -0.647f)),
    dot(v, float3(1.0f, -1.106f, 1.703f)));
}

float3 linearToSrgb(float3 c) {
  const float3 nonNegative = max(c, 0.0f);
  const float3 low = 12.92f * nonNegative;
  const float3 high = 1.055f * pow(nonNegative, 1.0f / 2.4f) - 0.055f;
  const float3 lowRegion = 1.0f - step(0.0031308f, nonNegative);
  return lerp(high, low, lowRegion);
}

float3 srgbToLinear(float3 c) {
  const float3 nonNegative = max(c, 0.0f);
  const float3 low = nonNegative / 12.92f;
  const float3 high = pow((nonNegative + 0.055f) / 1.055f, 2.4f);
  const float3 lowRegion = 1.0f - step(0.04045f, nonNegative);
  return lerp(high, low, lowRegion);
}

float3 sampleColor(float2 uv) {
  return InColorSampler.SampleLevel(uv, 0.0f).rgb;
}

float3 sampleDisplayColor(float2 uv) {
  return linearToSrgb(sampleColor(uv));
}

float sampleLuma(float2 uv) {
  return dot(sampleColor(uv), LUMA);
}

float sampleDisplayLuma(float2 uv) {
  return dot(sampleDisplayColor(uv), LUMA);
}

float frameNumber() {
  return floor(cb.time * 29.97f) + float(cb.frameIdx & 0xffffu);
}

float fieldNumber() {
  return floor(cb.time * 59.94f) + float(cb.frameIdx & 0xffffu) * 2.0f;
}

float lowpassWeight(int offset, float cutoffHz, int taps) {
  const float half = 0.5f * float(taps - 1);
  const float n = float(offset);
  const float normalizedCutoff = saturate(cutoffHz / NYQUIST);
  const float windowIndex = float(offset) + half;
  const float window = 0.54f - 0.46f * cos(TAU * windowIndex / float(taps - 1));
  const float sinc = abs(n) < 1e-4f
      ? normalizedCutoff
      : sin(PI * normalizedCutoff * n) / (PI * n);
  return sinc * window;
}

float lumaCutoffHz() {
  return max(cb.lumaBW, 0.25f) * 1e6f;
}

float colorCutoffHz() {
  return max(cb.colorBW, 1.0f) * 1e3f;
}

float3 applyVhsTapePath(float2 uv) {
  const float dx = 1.0f / ACTIVE_SAMPLES;
  float sumY = 0.0f;
  float sumI = 0.0f;
  float sumQ = 0.0f;
  float normY = 0.0f;
  float normC = 0.0f;

  [loop]
  for (int k = -30; k <= 30; ++k) {
    const float2 sampleUv = float2(uv.x + float(k) * dx, uv.y);
    const float3 yiq = toYIQ(sampleDisplayColor(sampleUv));
    const float yWeight = lowpassWeight(k, lumaCutoffHz(), 61);
    const float cWeight = lowpassWeight(k, colorCutoffHz(), 61);
    sumY += yiq.x * yWeight;
    sumI += yiq.y * cWeight;
    sumQ += yiq.z * cWeight;
    normY += yWeight;
    normC += cWeight;
  }

  float y = sumY / max(normY, 1e-4f);
  const float i = sumI / max(normC, 1e-4f);
  const float q = sumQ / max(normC, 1e-4f);

  float lowpassY = 0.0f;
  float normRing = 0.0f;
  [loop]
  for (int k = -15; k <= 15; ++k) {
    const float weight = lowpassWeight(k, 1.5e6f, 31);
    lowpassY += sampleDisplayLuma(float2(uv.x + float(k) * dx, uv.y)) * weight;
    normRing += weight;
  }
  lowpassY /= max(normRing, 1e-4f);
  y += cb.ringing * (y - lowpassY);

  return toRGB(float3(y, i, q));
}

float headSmearEnvelope(float line, float frame) {
  const float phase1 = hash(float2(frame + 11.0f, 17.0f)) * TAU;
  const float phase2 = hash(float2(frame + 23.0f, 29.0f)) * TAU;
  const float phase3 = hash(float2(frame + 37.0f, 41.0f)) * TAU;
  const float envelope =
      0.5f * sin(TAU * line / 73.0f + phase1) +
      0.3f * sin(TAU * line / 127.0f + phase2) +
      0.2f * sin(TAU * line / 41.0f + phase3);
  return clamp(envelope, 0.0f, 1.0f);
}

float gaussianNoiseSample(float sample, float line, float field) {
  const float u1 = max(hash3(float3(sample, line, field)), 1e-5f);
  const float u2 = hash3(float3(sample + 31.0f, line + 17.0f, field + 7.0f));
  return sqrt(-2.0f * log(u1)) * cos(TAU * u2);
}

float bandLimitedNoise(float sample, float line, float field) {
  float result = 0.0f;
  float variance = 0.0f;
  [loop]
  for (int k = -15; k <= 15; ++k) {
    const float weight = lowpassWeight(k, 3.0e6f, 31);
    result += weight * gaussianNoiseSample(floor(sample) + float(k), line, field);
    variance += weight * weight;
  }
  return result / sqrt(max(variance, 1e-5f));
}

float3 applyTapeStages(float2 uv) {
  const float dx = 1.0f / ACTIVE_SAMPLES;
  const float line = floor(uv.y * 480.0f);
  const float frame = frameNumber();
  const float field = fieldNumber();

  float3 c = sampleColor(uv);
  const float3 chroma = c - dot(c, LUMA);
  float y = dot(c, LUMA);

  const int maxRadius = (int)ceil(saturate(cb.headSmear) * 32.0f * ACTIVE_SAMPLES / 754.0f);
  const int radius = (int)round(headSmearEnvelope(line, frame) * float(maxRadius));
  if (radius > 0) {
    float sum = 0.0f;
    float count = 0.0f;
    [loop]
    for (int k = -32; k <= 32; ++k) {
      if (abs(k) <= radius) {
        sum += sampleLuma(float2(uv.x + float(k) * dx, uv.y));
        count += 1.0f;
      }
    }
    y = sum / max(count, 1.0f);
  }
  c = float3(y, y, y) + chroma;

  const float lumaLevel = saturate(y);
  const float sample = uv.x * ACTIVE_SAMPLES;
  y += cb.lumaNoise * (1.0f - 0.7f * lumaLevel)
     * bandLimitedNoise(sample, line, field);
  return float3(y, y, y) + chroma;
}

float dropoutLengthSamples(float line, float frame) {
  const float u = max(hash(float2(line + 71.0f, frame + 73.0f)), 1e-4f);
  return max(4.0f, -log(u) * cb.dropoutLengthUs * 14.31818f);
}

bool isDropout(float line, float frame, out float start, out float length) {
  const float rate = max(cb.dropoutRate, 0.0f);
  const float event = hash(float2(line + 47.0f, frame + 53.0f));
  start = hash(float2(line + 59.0f, frame + 61.0f)) * ACTIVE_SAMPLES;
  length = dropoutLengthSamples(line, frame);
  return rate > 0.0f && event < min(rate / 480.0f, 1.0f);
}

float3 applyTapeDropout(float2 uv) {
  const float line = floor(uv.y * 480.0f);
  const float frame = frameNumber();
  float start = 0.0f;
  float length = 0.0f;
  const float x = uv.x * ACTIVE_SAMPLES;
  if (!isDropout(line, frame, start, length) || x < start || x >= start + length) {
    return sampleColor(uv);
  }

  if (length < 20.0f) {
    const float spike = 0.6857f + 0.4f + hash(float2(line + 67.0f, frame + 79.0f)) * 0.25f;
    return float3(spike, spike, spike);
  }

  const float previousY = max(uv.y - 1.0f / 480.0f, 0.0f);
  return sampleColor(float2(uv.x, previousY));
}

float3 applyTapeTrail(float2 uv) {
  const float dx = 1.0f / ACTIVE_SAMPLES;
  const float alpha = 1.0f - saturate(cb.tapeTrail) * 0.70f;
  const float decay = 1.0f - alpha;
  float trailY = 0.0f;
  float weight = alpha;

  [unroll]
  for (int k = 0; k <= 32; ++k) {
    trailY += weight * sampleLuma(float2(uv.x - float(k) * dx, uv.y));
    weight *= decay;
  }

  const float3 c = sampleColor(uv);
  const float3 chroma = c - dot(c, LUMA);
  return float3(trailY, trailY, trailY) + chroma;
}

[shader("compute")]
[numthreads(NTSC_VHS_TILE_SIZE, NTSC_VHS_TILE_SIZE, 1)]
void main(uint2 pixelPos : SV_DispatchThreadID) {
  if (any(pixelPos >= cb.imageSize)) {
    return;
  }

  const float2 uv = (float2(pixelPos) + 0.5f) * cb.invImageSize;
  if (cb.pass == 0u) {
    OutColorTexture[pixelPos] = float4(applyVhsTapePath(uv), 1.0f);
  } else if (cb.pass == 1u) {
    OutColorTexture[pixelPos] = float4(applyTapeStages(uv), 1.0f);
  } else if (cb.pass == 2u) {
    OutColorTexture[pixelPos] = float4(applyTapeDropout(uv), 1.0f);
  } else {
    const float3 display = saturate(applyTapeTrail(uv));
    OutColorTexture[pixelPos] = float4(srgbToLinear(display), 1.0f);
  }
}
