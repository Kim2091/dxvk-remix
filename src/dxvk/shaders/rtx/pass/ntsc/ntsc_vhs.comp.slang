/*
 * NTSC/VHS composite post-process.
 *
 * The pass order follows Kim2091/ntsc-simulator's SignalEffects::apply:
 *
 *   pass 0: VHS luma/color bandwidth -> edge ringing
 *   pass 1: head-wear smear -> luma-dependent noise -> tape dropout
 *   pass 2: causal tape trail (last)
 *
 * The shader uses a fixed 640x480 active-picture grid. It is an image-space
 * approximation of the simulator's 4x-subcarrier signal path, but keeps the
 * important semantics: VHS operations act on luma/chroma separately, noise
 * gets darker-scene weighting, dropouts are sparse horizontal events, and the
 * tape trail is a causal luma-only IIR.
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
static const float NW = 640.0f;
static const float NH = 480.0f;

float hash(float2 v) {
  return frac(sin(dot(v, float2(12.9898f, 78.233f))) * 43758.5453f);
}

float hash3(float3 v) {
  return frac(sin(dot(v, float3(12.9898f, 78.233f, 37.719f))) * 43758.5453f);
}

float3 toYIQ(float3 c) {
  return float3(
    dot(c, LUMA),
    dot(c, float3(0.596f, -0.274f, -0.322f)),
    dot(c, float3(0.211f, -0.523f, 0.312f)));
}

float3 toRGB(float3 v) {
  return float3(
    dot(v, float3(1.0f, 0.956f, 0.621f)),
    dot(v, float3(1.0f, -0.272f, -0.647f)),
    dot(v, float3(1.0f, -1.106f, 1.703f)));
}

float3 sampleColor(float2 uv) {
  return InColorSampler.SampleLevel(uv, 0.0f).rgb;
}

float sampleLuma(float2 uv) {
  return dot(sampleColor(uv), LUMA);
}

float frameNumber() {
  return floor(cb.time * 29.97f) + float(cb.frameIdx & 0xffffu);
}

float fieldNumber() {
  return floor(cb.time * 59.94f) + float(cb.frameIdx & 0xffffu) * 2.0f;
}

// The Rust simulator's extra VHS luma filter is a 3 MHz SP filter. These
// triangular windows are expressed in fixed active-picture samples rather
// than output pixels, so the same tuning behaves at different resolutions.
float lumaHalfWidth() {
  return max(1.0f, 3.0f * 3.0f / max(cb.lumaBW, 0.25f));
}

float colorHalfWidth() {
  return lerp(20.0f, 8.0f, saturate((cb.colorBW - 300.0f) / 200.0f));
}

float linePhase(float line, float frame) {
  // 227.5 subcarrier cycles per NTSC line become a half-cycle line phase
  // after the 4x-sample representation is projected to the active grid.
  return line * PI + frame * PI;
}

float3 ntscDecode(float2 uv) {
  const float du = 1.0f / NW;
  const float line = floor(uv.y * NH);
  const float frame = frameNumber();
  const float fsc = 0.5f;
  const float yHalf = lumaHalfWidth();
  const float cHalf = colorHalfWidth();

  float sumY = 0.0f;
  float sumI = 0.0f;
  float sumQ = 0.0f;
  float normY = 0.0f;
  float normC = 0.0f;

  [unroll]
  for (int k = -24; k <= 24; ++k) {
    const float fk = float(k);
    const float2 sampleUv = float2(uv.x + fk * du, uv.y);
    const float3 yiq = toYIQ(sampleColor(sampleUv));
    const float phase = PI * 0.5f * fsc * (uv.x * NW + fk) + linePhase(line, frame);
    const float composite = yiq.x + yiq.y * cos(phase) + yiq.z * sin(phase);
    const float yWeight = max(0.0f, 1.0f - abs(fk) / yHalf);
    const float cWeight = max(0.0f, 1.0f - abs(fk) / cHalf);

    sumY += composite * yWeight;
    sumI += composite * cos(phase) * cWeight;
    sumQ += composite * sin(phase) * cWeight;
    normY += yWeight;
    normC += cWeight;
  }

  // Multiplication by two is the demodulator gain used by the Rust path.
  float3 decoded = float3(sumY / max(normY, 1e-4f),
                          2.0f * sumI / max(normC, 1e-4f),
                          2.0f * sumQ / max(normC, 1e-4f));

  // Rust: luma + ringing * (luma - lowpass(luma)). Do not multiply the
  // requested strength by another hidden gain.
  float lowpassY = 0.0f;
  [unroll]
  for (int k = -3; k <= 3; ++k) {
    lowpassY += sampleLuma(float2(uv.x + float(k) * du, uv.y));
  }
  lowpassY /= 7.0f;
  decoded.x += cb.ringing * (decoded.x - lowpassY);

  return toRGB(decoded);
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

float gaussianNoise(float2 pixel, float field) {
  // Four uniform samples approximate the unit-variance normal used by Rust.
  const float s =
      hash3(float3(pixel + float2(-1.0f, 0.0f), field)) +
      hash3(float3(pixel + float2(0.0f, -1.0f), field + 7.0f)) +
      hash3(float3(pixel + float2(1.0f, 0.0f), field + 13.0f)) +
      hash3(float3(pixel + float2(0.0f, 1.0f), field + 19.0f));
  return (s - 2.0f) * 1.7320508f;
}

float dropoutLengthPixels(float line, float frame) {
  const float u = max(hash(float2(line + 71.0f, frame + 73.0f)), 1e-4f);
  // 4x FSC is 14.318 samples/us; 640 active pixels represent ~754 samples.
  return max(4.0f * NW / 754.0f,
             -log(u) * cb.dropoutLengthUs * 14.31818f * NW / 754.0f);
}

bool isDropout(float line, float frame, out float start, out float length) {
  const float rate = max(cb.dropoutRate, 0.0f);
  const float event = hash(float2(line + 47.0f, frame + 53.0f));
  start = hash(float2(line + 59.0f, frame + 61.0f));
  length = dropoutLengthPixels(line, frame);
  return rate > 0.0f && event < min(rate / NH, 1.0f);
}

float3 applyTapeStages(float2 uv) {
  const float du = 1.0f / NW;
  const float dv = 1.0f / NH;
  const float line = floor(uv.y * NH);
  const float frame = frameNumber();
  const float field = fieldNumber();

  float3 c = sampleColor(uv);
  const float3 chroma = c - dot(c, LUMA);
  float y = dot(c, LUMA);

  // Rust head-wear smear: variable-width symmetric box blur on luma only.
  const int maxRadius = (int)ceil(saturate(cb.headSmear) * 32.0f * NW / 754.0f);
  const int radius = (int)round(headSmearEnvelope(line, frame) * float(maxRadius));
  if (radius > 0) {
    float sum = 0.0f;
    float count = 0.0f;
    [loop]
    for (int k = -32; k <= 32; ++k) {
      if (abs(k) <= radius) {
        sum += sampleLuma(float2(uv.x + float(k) * du, uv.y));
        count += 1.0f;
      }
    }
    y = sum / max(count, 1.0f);
    c = float3(y, y, y) + chroma;
  }

  // Rust tape noise is darker-scene weighted: 1.0 at black, 0.3 at white.
  const float lumaLevel = saturate(y / 0.66f);
  y += cb.lumaNoise * (1.0f - 0.7f * lumaLevel)
     * gaussianNoise(float2(floor(uv.x * NW), line), field);
  c = float3(y, y, y) + chroma;

  // Rust dropouts are sparse line events, with exponential length and a
  // previous-line compensator for the usual long dropout.
  float start = 0.0f;
  float length = 0.0f;
  if (isDropout(line, frame, start, length) && uv.x >= start && uv.x < start + length) {
    if (length < 20.0f * NW / 754.0f) {
      const float spike = 0.2857f + 0.4f + hash(float2(line + 67.0f, frame + 79.0f)) * 0.25f;
      y = spike;
      c = float3(y, y, y) + chroma;
    } else {
      c = sampleColor(float2(uv.x, uv.y - dv));
    }
  }

  return saturate(c);
}

float3 applyTapeTrail(float2 uv) {
  const float du = 1.0f / NW;
  const float alpha = 1.0f - saturate(cb.tapeTrail) * 0.70f;
  const float decay = 1.0f - alpha;
  float trailY = 0.0f;
  float weight = alpha;

  // This is the finite form of y[n] = alpha*x[n] + (1-alpha)*y[n-1].
  [unroll]
  for (int k = 0; k <= 32; ++k) {
    trailY += weight * sampleLuma(float2(uv.x - float(k) * du, uv.y));
    weight *= decay;
  }

  const float3 c = sampleColor(uv);
  const float3 chroma = c - dot(c, LUMA);
  return saturate(float3(trailY, trailY, trailY) + chroma);
}

[shader("compute")]
[numthreads(NTSC_VHS_TILE_SIZE, NTSC_VHS_TILE_SIZE, 1)]
void main(uint2 pixelPos : SV_DispatchThreadID) {
  if (any(pixelPos >= cb.imageSize)) {
    return;
  }

  const float2 uv = (float2(pixelPos) + 0.5f) * cb.invImageSize;
  float3 result;
  if (cb.pass == 0u) {
    result = saturate(ntscDecode(uv));
  } else if (cb.pass == 1u) {
    result = applyTapeStages(uv);
  } else {
    result = applyTapeTrail(uv);
  }

  OutColorTexture[pixelPos] = float4(result, 1.0f);
}
