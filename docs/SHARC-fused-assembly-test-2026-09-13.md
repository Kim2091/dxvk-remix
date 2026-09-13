# FNV fused assembly test candidate ? 2026-09-13

Code commit: 4b3b48fe7 (local only).

Installed Release d3d9.dll and matching d3d9.pdb in FNV's .trex directory. Source and deployed SHA-256 hashes matched:
- DLL: FF1E1784CE3F54C2A323890858B98591F817C636EC31ED80778E81C8EDADB063
- PDB: C24B00AD276B2CFA404F1DF43C101AB94D022B4077CD8DA9D10ED66EECB7E55B

Rollback directory: `C:\Users\sparkles\Projects\Games\Fallout New Vegas\.trex\.backup-pre-fused-assembly-20260913-061838`. Contains previous DLL/PDB, rtx.conf, and prior runtime log. Root bridge, bridge server, and SDK DLLs were preserved. Only rtx.sharc.fuseAssembly = True was added to the game configuration.

FNV was launched through nvse_loader.exe; FalloutNV and NvRemixBridge processes were observed running. Startup Vulkan initialization succeeded. The runtime logged [SHARC] Combined query/assembly active at 06:19:00, confirming effective selection. Visual correctness and comparative frame timings remain pending user testing. No measured performance benefit is claimed.

Load the same save/view. Inspect emissive lights, glass/reflections, and camera movement. Then hold the camera still for 45?60 seconds. The runtime activation marker is [SHARC] Combined query/assembly active. Compare combined IndirectIntegration + IndirectAssembly and overall frame time against standalone; assembly cost moves into integration in this candidate.

For a baseline test, exit FNV, set rtx.sharc.fuseAssembly = False, and relaunch the same build. To fully restore the previous installation, exit FNV and restore both DLL/PDB plus rtx.conf from the rollback directory.

## First capture and same-build baseline follow-up

User completed the test; no explicit visual-quality report was supplied. Preserved runtime log in _Comp64Release/fused-assembly-first-capture.log and settings in fused-assembly-first-capture-rtx.conf.

Last 12 complete samples, fused frames 3298?4618 versus previous split frames 2534?3854:
- Median per-frame sum of IndirectIntegration + IndirectAssembly: split 2.682511 ms, fused 2.955839 ms (+0.273328 ms, approximately 10.2%).
- Median measured GPU sequence: split 15.932 ms, fused 15.628 ms.
- Samples vary substantially; these captures do not establish a causal speedup or definitive regression. The target interval points against fusion despite a lower overall median.
- GPU sequence excludes game CPU and presentation work.

With FNV closed, retained the same DLL/PDB and changed only rtx.sharc.fuseAssembly to False for a fresh standalone baseline. Restarting with false restores the original aliased throughput allocation; this compares the full enabled/disabled feature cost. A same-allocation live toggle comparison has not been performed. Next test: same save/view, stationary 45?60 seconds, with visual feedback.
