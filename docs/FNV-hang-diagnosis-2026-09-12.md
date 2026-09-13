# FNV live hang diagnosis - 2026-09-12

User clarified this incident was a hang, not a crash. FNV PID 44404 (started
21:45:04) was nonresponsive; NvRemixBridge PID 47892 (started 21:45:09) remained
running. Noninvasive CDB attachments read stacks and memory and then detached.
No game processes were terminated and no live memory was changed.

## Confirmed blocking condition

Both bridge command rings are full, producing a circular wait:
- Server ring at 0x000001fccee75440: capacity 10, consumer index 0,
  producer index 9 (9 usable entries occupied). Server main thread is in the
  push loop, attempting command 5 (Bridge_Response), UID 0x22c7e176.
- Client ring at 0x01949030: capacity 0xc00 (3072), consumer index 0x577,
  producer index 0x576 (3071 usable entries occupied). Game main thread is
  spinning in the bridge client's push loop.
- Repeated stack samples show the same blocking loops. Renderer dxvk-cs,
  dxvk-submit and dxvk-queue threads are waiting on work condition variables.
  This evidence supports a bridge IPC deadlock, not an executing SHARC shader
  or GPU-completion wait as the immediate cause.

The initiating response-consumption error is not yet identified. Increasing
queue size alone would only postpone recurrence if responses keep accumulating.
Do not discard queued messages or patch live indices: that breaks protocol state.

## Deployment details

Both live bridge logs report remix-main+d3431ebd. Their binaries differ from
the bridge binaries in the recently prepared package. This does NOT establish
that client and server are incompatible with each other.
FNV root d3d9.dll is a custom wrapper; the actual bridge client is d3d9_remix.dll.
Preserve that wrapper if deploying a bridge pair.
Deployed bridge server SHA256:
AA47B8269C889FF4AD227316C9A7ACDDE03260A12FA868C91CB484124BEDACED
Deployed d3d9_remix.dll SHA256:
59BD4E8FF1120890D429DE8284700CA9975677225D22472C67F6F23B12A19FAD

Bridge PDB labels are misleading for some live addresses (e.g. mkDirs labels
on an atomic ring push loop). Queue diagnosis uses actual machine instructions
and explicit memory reads, not those function labels.

## Separate findings

The 21:44:47 FalloutNV.exe.12800.dmp concerns a different process, only 10 seconds
old. It records FAST_FAIL_INVALID_ARG in NVSE ErrorLogHook -> _vfprintf_l while
logging "Loading a user created or misnamed BSA". It is not evidence about the
current live hang. Earlier 21:17 NVSE access violation also concerns another run.

Screenshot exposes an independent SHARC UI printf bug: the literal 25% in
ImGui::TextWrapped was treated as a conversion (percent-space-o). Fixed by passing
the text through a %s format. Release build succeeded; not deployed while FNV is
running. No shader changes, no new shader validation required for this UI fix.

User reports update cache resampling gives no noticeable speed benefit. Screenshot:
update 0.39 ms, resolve 0.02 ms, query 1.30 ms, statistics ON; TraceRay+SER and WBOIT.
Update resampling targets only update time; prioritize query costs for further
performance work. Leave resampling optional/off for normal use.

## Evidence

Files under _Comp64Release:
- fnv-live-hang-stacks.txt
- fnv-bridge-live-hang-stacks.txt
- fnv-bridge-wait-detail.txt
- fnv-server-queue-state.txt / fnv-client-queue-state.txt
- fnv-server-queue-indices.txt / fnv-client-queue-indices.txt
- fnv-crash-20260912-214446.txt / fnv-crash-errorlog-details.txt
- sharc-ui-format-fix-build.log

Next investigation: identify the server responses that are not consumed, tracing
UIDs through the deployed bridge/custom wrapper request and response paths.
A matched current bridge pair is a reasonable controlled experiment, not a proven
fix. Do not overwrite the root custom wrapper with the bridge client.

## Follow-up source and binary review

Offline disassembly of the deployed x86 d3d9_remix.dll identifies the game
main-thread caller at RVA 0x34ffa as SetRenderState: it updates the render-state
array, sends command 0x6e with two DWORDs, then conditionally waits for an optional
response. It is blocked sending that command, before the optional response wait.
No other captured FNV stack contains d3d9_remix, so the snapshot does not show a
second client response consumer actively waiting. Accumulated abandoned responses
remain plausible; concurrent waiters are not proven as the incident trigger.

The first Terra candidate built but review found ABBA lock inversion between
its separate nonrecursive response mutex and the existing device mutex at Reset,
ResetEx and EndStateBlock. It was NOT deployed. Rework and regression verification
are in progress. Root review also identified nested readback handling that calls
surface methods before popping the current response; recursion alone does not
make nested outstanding response transactions safe.

## Revised candidate

ResponseTransaction now shares a single client-wide recursive mutex with the
multithreaded device guard. Per-request scopes cover command send through response
consumption, including queries and Remix API game-value requests. Using the same
mutex removes the first candidate's device/response lock-order inversion.

A mismatched response UID or exhausted response timeout disables further bridge
communication with an explicit error; subsequent waits return immediately.
Reset/ResetEx return DEVICELOST for a missing response and propagate failed reset
results instead of rebuilding implicit objects or reporting success. Surface
readback copies response bytes and consumes its header before uploading through
UnlockRect, so an outgoing upload cannot retain that response queue slot.

Validation includes the existing atomic-queue smoke test, common-lock exclusion
and recursion, and a new paired-queue test: two callers issue 500 requests each
through production AtomicCircularQueue pairs and ResponseTransaction. A simulated
server echoes UIDs; each caller checks exact response identity and both rings must
finish empty. Meson enforces a five-second process timeout. This verifies the
serialization mechanism, not every D3D9 call path or the unknown original trigger.
The x86 client and x64 server release builds succeeded. Test result and deployment
are pending. UI text formatting fix is included in the renderer build separately.

## Validation and deployment completed (2026-09-12 23:09 local)

- x86 bridge client and x64 server release builds passed.
- Meson command_history regression passed in 0.17 seconds (5-second timeout), including 1,000 paired concurrent request/response UIDs and recursive common-lock exclusion.
- git diff --check passed.
- Deployed bridge client, server, renderer UI formatting correction, and matching symbols with SHA256 verification. Game and bridge were closed.
- Root custom d3d9.dll wrapper was preserved and its hash verified. Client symbols are in .trex/bridge-client-symbols; wrapper symbols were not replaced.
- Deployment record: _Comp64Release/fnv-bridge-hangfix-deployment-20260912-230900.json
- Backups use suffix .backup-pre-bridge-hangfix-20260912-230900.

The captured two-full-ring deadlock is confirmed. The initiating event remains unknown; these changes address response ordering, abandoned responses, and unsafe reset/readback handling. The unit test does not prove the original FNV hang is eliminated. Runtime validation remains pending: repeat normal gameplay and the actions that previously preceded hangs. If it hangs again, leave both processes running for noninvasive stack and queue capture.

C:\Users\sparkles\Projects\Games\Fallout New Vegas\d3d9_remix.dll: 20A44182FF325E1580BF23969DA656DB9BAABC3FF659C7516AB50CC4015EC162
C:\Users\sparkles\Projects\Games\Fallout New Vegas\.trex\bridge-client-symbols\d3d9.pdb: D15A2FA78B0A17F180038077ECB2A5576983F1168430C10CFB0C099A7815E8C3
C:\Users\sparkles\Projects\Games\Fallout New Vegas\.trex\NvRemixBridge.exe: BC0E88EEC6D78B019BF8FEF203977CD18FC69505161B91E050D536296BB6ABD6
C:\Users\sparkles\Projects\Games\Fallout New Vegas\.trex\NvRemixBridge.pdb: E78971E12F49772991DD12072AB95C63E3BEEA10DC35C710BF9D165FE14BF9B3
C:\Users\sparkles\Projects\Games\Fallout New Vegas\.trex\d3d9.dll: 0CCC2F648BE153C2D5E5B26BC3612A24EFC6B1014E0D750E89B2DC77660D3DD3
C:\Users\sparkles\Projects\Games\Fallout New Vegas\.trex\d3d9.pdb: 1976371B72DDACF91E0FE295FFF657E5CB60607DF8448E9180B0A3DB9629625B
