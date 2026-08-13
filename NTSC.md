# NTSC/VHS composite post-process

The post-process is based on the signal path in
[Kim2091/ntsc-simulator](https://github.com/Kim2091/ntsc-simulator).
It runs after tonemapping on the final LDR image.

The stages are deliberately ordered like SignalEffects::apply in the Rust
implementation:

1. Direct sRGB/YIQ VHS luma and color-under bandwidth. The render target is
   not a 4x-subcarrier composite waveform, so synthesizing one at 640 pixels
   creates exaggerated rainbow/dot-crawl artifacts; the direct YIQ path is the
   equivalent decoded-image operation.
2. Playback edge ringing.
3. Luma-only worn-head smear.
4. Luminance-dependent tape noise.
5. Sparse tape dropouts with a previous-line compensator.
6. Causal luma-only tape trail as the final stage.

The implementation uses four GPU passes so dropout compensation reads the
already smeared/noisy previous line. The tape path runs in sRGB/YIQ space and
the final trail pass converts back to the linear post-tonemap render target.

Enable it with:

~~~ini
rtx.ntsc.ntscEnable = True
~~~

The defaults correspond to the midpoint of the requested ranges:

~~~ini
rtx.ntsc.ntscLumaBW             = 3.0
rtx.ntsc.ntscColorBW            = 425.0
rtx.ntsc.ntscRinging            = 0.30
rtx.ntsc.ntscLumaNoise          = 0.025
rtx.ntsc.ntscTapeDropoutRate    = 0.50
rtx.ntsc.ntscHeadSmear          = 0.175
rtx.ntsc.ntscTapeTrail          = 0.675
~~~

ntscTapeDropoutLength defaults to the Rust simulator's 15 microsecond
average. The runtime options are scalar values; the Rust CLI's min..max
syntax samples once per input file, so the midpoint is used for the live
runtime defaults.
