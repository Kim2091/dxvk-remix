/*
 * NTSC/VHS composite post-process - shared shader/C++ header.
 * The fields intentionally mirror the Rust simulator's tape-path controls.
 */
#pragma once

#define NTSC_VHS_INPUT   0   // Sampler2D  : composited LDR color (read)
#define NTSC_VHS_OUTPUT  1   // RWTexture2D: processed output (write)

#define NTSC_VHS_TILE_SIZE 8

struct NtscVhsArgs {
  uint2  imageSize;
  float2 invImageSize;

  float  time;              // seconds
  float  lumaBW;            // MHz (VHS tape luma bandwidth)
  float  colorBW;           // kHz (VHS color-under bandwidth)
  float  ringing;           // playback peaking gain

  float  lumaNoise;         // luminance-dependent tape noise amplitude
  float  dropoutRate;       // average dropouts per 480-line frame
  float  dropoutLengthUs;   // average dropout length in microseconds
  float  headSmear;         // worn-head symmetric luma smear strength

  float  tapeTrail;         // causal luma IIR strength, 0..1
  uint   frameIdx;           // fallback/deterministic animation seed
  uint   pass;               // 0 = NTSC, 1 = tape stages, 2 = tape trail
  float  _pad1;
};
