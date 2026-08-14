# NTSC/VHS composite post-process

The NTSC/VHS effect is a display-space stack member. It runs after tonemapping
and before the final sRGB conversion/dither pass, so it can be reordered with
other display-space effects without crossing the HDR-to-display boundary.

The implementation follows the tape path used by
[Kim2091/ntsc-simulator](https://github.com/Kim2091/ntsc-simulator), adapted to
a normal rendered image rather than a sampled 4x-subcarrier waveform:

1. sRGB/YIQ luma and color-under bandwidth reduction with playback ringing;
2. worn-head luma smear and luminance-dependent tape noise;
3. sparse tape dropouts with previous-line compensation;
4. causal luma-only tape trail and conversion back to linear RGB.

The effect uses four compute passes because dropout compensation must read the
already smeared/noisy result. It is disabled by default:

```ini
rtx.ntsc.ntscEnable = True
```

The main controls are `ntscLumaBW`, `ntscColorBW`, `ntscRinging`,
`ntscLumaNoise`, `ntscTapeDropoutRate`, `ntscTapeDropoutLength`,
`ntscHeadSmear`, and `ntscTapeTrail` in the `rtx.ntsc` category. The defaults
use a 3 MHz luma path, 425 kHz color-under path, moderate ringing/noise, and
the simulator's 15 microsecond average dropout length.
