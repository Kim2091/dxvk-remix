/*
 * NTSC/VHS composite post-process - shared shader/C++ header.
 * The fields intentionally mirror the Rust simulator's tape-path controls.
 */
#pragma once

#define NTSC_VHS_INPUT   0   // Sampler2D  : post-tonemap linear/sRGB color (read)
#define NTSC_VHS_OUTPUT  1   // RWTexture2D: processed color (write)

#define NTSC_VHS_TILE_SIZE 8

struct NtscVhsArgs {
  uint2  imageSize;
  float2 invImageSize;

  float  time;
  float  lumaBW;
  float  colorBW;
  float  ringing;

  float  lumaNoise;
  float  dropoutRate;
  float  dropoutLengthUs;
  float  headSmear;

  float  tapeTrail;
  uint   frameIdx;
  uint   pass;
  float  _pad1;
};
