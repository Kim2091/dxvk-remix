#!/bin/bash
# D3D9 v2 synthetic scenarios: every remix_testhost --d3d9 scenario once on the old D3D9 path ("off") and once with
# every D3D9 v2 option on ("on"), then PASS / FAIL per expectation from the runtime logs.
#
#   RT=<runtime folder> ./d3d9v2_scenarios.sh [scenario ...]      (default: all scenarios)
#
# RT    a runtime folder holding all of _output (d3d9.dll and the rest), as for p5gates R4.
# BIN   the folder with remix_testhost.exe (default: bin next to this script).
# OUT   where the logs go (default: bin/scenarios next to this script).
# WARMUP 1 (default): one unrecorded run per mode first, so the shader pipelines are compiled and cached before the
#        first recorded run (a cold cache can leave a whole run without a ray traced interval).
# FRAMES frames per run (default 720: the [D3D9v2] and [SceneStats] lines come every 300 frames, so the last full
#        interval, frames 300-599, holds each scenario's event at frame 450).
#
# The testhost has no --set, so the options go into $RT/user.conf (it wins over rtx.conf), which is restored on exit.
# About 16 scenarios x 2 runs x 30-40 s. It does not start anything else: build the testhost and deploy the runtime first.

HERE=$(cd "$(dirname "$0")" && pwd)
RT=${RT:?set RT to a runtime folder (all of _output), for example RT=/c/path/to/replay-runtime}
BIN=${BIN:-$HERE/bin}
OUT=${OUT:-$HERE/bin/scenarios}
FRAMES=${FRAMES:-720}
COOL=${COOL:-10}
WARMUP=${WARMUP:-1}
TIMEOUT=${TIMEOUT:-600}
ALL="repeat twice twice-overlay twice-overlay-first worldspace skinned identity-view fused-world fused-view reset no-present sky terrain emitter portals rendertarget vtf"
SCENARIOS=${@:-$ALL}
TH="$BIN/remix_testhost.exe"
LOG="$RT/rtx-remix/logs/remix-dxvk.log"

[ -x "$TH" ] || { echo "missing $TH (build tools/remix_replay/build.bat)"; exit 3; }
[ -f "$RT/d3d9.dll" ] || { echo "missing $RT/d3d9.dll"; exit 3; }
mkdir -p "$OUT"

busy() {
  for exe in Fallout4.exe hl2.exe NvRemixBridge.exe remix_replay.exe remix_testhost.exe; do
    if tasklist //FI "IMAGENAME eq $exe" 2>/dev/null | grep -qi "${exe%.exe}"; then echo "$exe"; return 0; fi
  done
  return 1
}

# user.conf: kept as user.conf.d3d9v2orig and restored on exit (removed again if there was none).
HAD_CONF=0
if [ -f "$RT/user.conf" ]; then HAD_CONF=1; cp "$RT/user.conf" "$RT/user.conf.d3d9v2orig"; else : > "$RT/user.conf.d3d9v2orig"; fi
restore() {
  if [ $HAD_CONF = 1 ]; then cp "$RT/user.conf.d3d9v2orig" "$RT/user.conf"; else rm -f "$RT/user.conf"; fi
  rm -f "$RT/user.conf.d3d9v2orig"
  echo "user.conf restored"
}
trap restore EXIT

# setconf on|off key=value...: the original user.conf plus the logging keys, every D3D9 v2 option (True for on, False
# for off) and the scenario's own keys.
setconf() {
  python - "$RT/user.conf.d3d9v2orig" "$RT/user.conf" "$@" <<'PY'
import sys
src, dst, mode = sys.argv[1:4]
on = mode == "on"
keys = {
    "rtx.d3d9v2.log": "True",
    "rtx.d3d9Identity.log": "True",
    "rtx.profile.detailedLogging": "True",
    "rtx.coreMaterial.selfCheck": "True",
    "rtx.scene.selfCheckFrames": "60",
    "rtx.d3d9v2.earlyChangeTestSelfCheckFrames": "60" if on else "0",
}
for k in ("scene.directRetained scene.directUnordered scene.directServices scene.d3d9CutAsEvent d3d9v2.direct "
          "d3d9v2.materialTranslator d3d9v2.explicitCamera d3d9v2.earlyChangeTest d3d9v2.directReplacements "
          "d3d9v2.firstDrawHandover d3d9v2.dynamicGeometry d3d9v2.capturedGeometry d3d9v2.unordered "
          "d3d9v2.specialCategories d3d9v2.skyService d3d9v2.terrain d3d9v2.skybox3d d3d9v2.renderTargets "
          "d3d9v2.unknownCamera d3d9v2.lights d3d9v2.frameEvents").split():
    keys["rtx." + k] = "True" if on else "False"
for kv in sys.argv[4:]:
    k, v = kv.split("=", 1)
    keys[k.strip()] = v.strip()
lines = [l for l in open(src).read().splitlines() if l.split("=")[0].strip() not in keys]
lines += [f"{k} = {v}" for k, v in keys.items()]
open(dst, "w", newline="\r\n").write("\n".join(lines) + "\n")
PY
}

# The texture hashes of the tagged scenarios (remix_testhost --d3d9-hashes).
while read -r tag name value; do
  [ "$tag" = hash ] && declare "H_$name=$value"
done < <("$TH" --d3d9-hashes | tr -d '\r')
[ -n "$H_terrain" ] && [ -n "$H_monitor" ] || { echo "remix_testhost --d3d9-hashes printed no hashes"; exit 3; }

# scenario NAME: sets ARGS (testhost) and CONF (user.conf keys for both modes).
scenario() {
  ARGS=(); CONF=()
  case $1 in
    repeat)        ARGS=(--d3d9 repeat); CONF=("rtx.antiCulling.object.enable=True") ;;
    twice)         ARGS=(--d3d9 twice) ;;
    twice-overlay) ARGS=(--d3d9 twice --overlay) ;;
    twice-overlay-first) ARGS=(--d3d9 twice --overlay-first) ;;
    worldspace)    ARGS=(--d3d9 worldspace) ;;
    skinned)       ARGS=(--d3d9 skinned) ;;
    identity-view) ARGS=(--d3d9 identity-view); CONF=("rtx.fusedWorldViewMode=0" "rtx.antiCulling.object.enable=True") ;;
    fused-world)   ARGS=(--d3d9 fused-world); CONF=("rtx.fusedWorldViewMode=2" "rtx.antiCulling.object.enable=True") ;;
    fused-view)    ARGS=(--d3d9 fused-view); CONF=("rtx.fusedWorldViewMode=1" "rtx.antiCulling.object.enable=True") ;;
    reset)         ARGS=(--d3d9 reset) ;;
    no-present)    ARGS=(--d3d9 no-present) ;;
    sky)           ARGS=(--d3d9 sky); CONF=("rtx.skyDrawcallIdThreshold=2") ;;
    terrain)       ARGS=(--d3d9 terrain); CONF=("rtx.terrainTextures=$H_terrain" "rtx.terrainBaker.enableBaking=True") ;;
    emitter)       ARGS=(--d3d9 emitter); CONF=("rtx.particleEmitterTextures=$H_emitter") ;;
    portals)       ARGS=(--d3d9 portals); CONF=("rtx.rayPortalModelTextureHashes=$H_portalA, $H_portalB") ;;
    rendertarget)  ARGS=(--d3d9 rendertarget); CONF=("rtx.raytracedRenderTargetTextures=$H_monitor" "rtx.raytracedRenderTarget.enable=True") ;;
    vtf)           ARGS=(--d3d9 vtf); CONF=("rtx.useVertexCapture=True") ;;
    *) echo "unknown scenario $1 (known: $ALL)"; return 1 ;;
  esac
}

if [ "$WARMUP" = 1 ]; then
  scenario twice
  for mode in off on; do
    if b=$(busy); then echo "$b is running before the warm-up, stopping"; exit 2; fi
    setconf $mode "${CONF[@]}"
    timeout $TIMEOUT "$TH" --runtime "$RT" --frames $FRAMES "${ARGS[@]}" > /dev/null 2>&1
    echo "=== warm-up $mode exit $?"
  done
fi

for sc in $SCENARIOS; do
  scenario $sc || exit 2
  for mode in off on; do
    name=$sc-$mode
    if b=$(busy); then echo "$b is running before $name, stopping"; exit 2; fi
    sleep $COOL
    setconf $mode "${CONF[@]}"
    rm -f "$LOG"
    timeout $TIMEOUT "$TH" --runtime "$RT" --frames $FRAMES "${ARGS[@]}" > "$OUT/out-$name.txt" 2>&1
    rc=$?
    if [ -f "$LOG" ]; then cp "$LOG" "$OUT/log-$name.txt"; else : > "$OUT/log-$name.txt"; fi
    echo "=== $name exit $rc"
    grep -E '^(event|no-present|SKIP|FAILED)' "$OUT/out-$name.txt" | head -4
  done
done

# The checks. Each line: PASS / FAIL / INFO / SKIP, the scenario, the checklist ID, what is expected and the values.
python - "$OUT" $SCENARIOS <<'PY'
import os, re, sys

out = sys.argv[1]
results = {"PASS": 0, "FAIL": 0}

def read(path):
    try:
        return open(path, errors="replace").read().splitlines()
    except OSError:
        return []

def val(line, key):
    if line is None:
        return None
    m = re.search(r"(?:^|[\s(|])" + re.escape(key) + r"=([^\s,)|]+)", line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return m.group(1)

class Log:
    def __init__(self, sc, mode):
        self.lines = read(os.path.join(out, f"log-{sc}-{mode}.txt"))
        self.out = read(os.path.join(out, f"out-{sc}-{mode}.txt"))
        self.done = any("frames done" in l for l in self.out)
        self.skip = next((l for l in self.out if l.startswith("SKIP")), None)
        # [D3D9v2] inputs intervals, each with the notEligible line printed right after it (none when empty).
        self.inputs = []
        for l in self.lines:
            if "[D3D9v2] inputs frames=" in l:
                self.inputs.append({"inputs": l, "ne": ""})
            elif "[D3D9v2] notEligible" in l and self.inputs:
                self.inputs[-1]["ne"] = l
        self.direct = [l for l in self.lines if "[D3D9v2] direct (totals over" in l]
        self.stats = [l for l in self.lines if "[SceneStats] tlasInstances=" in l]
        self.identity = [l for l in self.lines if "[D3d9Identity] frames=" in l]
        self.sceneCheck = sum(1 for l in self.lines if "[SceneCheck] frame" in l)
        self.preKeyMismatch = sum(int(val(i["inputs"], "preKeyMismatch") or 0) for i in self.inputs)

    def lastInputs(self, key):
        return val(self.inputs[-1]["inputs"], key) if self.inputs else None

    def lastNotEligible(self):
        if not self.inputs:
            return None
        ne = self.inputs[-1]["ne"]
        return ne.split("it is in):", 1)[-1].strip() if ne else ""

    def lastDirect(self, key):
        return val(self.direct[-1], key) if self.direct else None

    def rejected(self):
        if not self.direct:
            return None
        m = re.search(r"\| rejected:(.*?) \| lights", self.direct[-1])
        return m.group(1).strip() if m else None

    def stat(self, key, which=-1):
        return val(self.stats[which], key) if self.stats else None

def report(status, sc, cid, text, values=""):
    if status in results:
        results[status] += 1
    print(f"{status:4} {sc:13} {cid:4} {text}" + (f"  [{values}]" if values else ""))

def expect(cond, sc, cid, text, values=""):
    if cond is None:
        report("FAIL", sc, cid, text, (values + "; " if values else "") + "counter missing from the log")
    else:
        report("PASS" if cond else "FAIL", sc, cid, text, values)

def num(x):
    return None if x is None else (int(x) if float(x).is_integer() else x)

def both(a, b, f):
    return None if a is None or b is None else f(a, b)

for sc in sys.argv[2:]:
    off, on = Log(sc, "off"), Log(sc, "on")
    if on.skip or off.skip:
        report("SKIP", sc, "-", (on.skip or off.skip))
        continue
    if not (off.done and on.done):
        report("FAIL", sc, "run", "both runs finish (testhost: ... frames done)", f"off {off.done}, on {on.done}")
        continue

    # Every scenario: the scene self-checks, coverage and the old-path reference.
    expect(on.sceneCheck == 0, sc, "SC", "on: no [SceneCheck] mismatch", f"{on.sceneCheck} lines")
    expect(on.preKeyMismatch == 0 if on.inputs else None, sc, "R1", "on: preKeyMismatch=0", f"{on.preKeyMismatch}")
    ne = on.lastNotEligible()
    expect(None if ne is None else ne == "", sc, "cov", "on: [D3D9v2] notEligible shows no class (last interval)", ne or "")
    rej = on.rejected()
    expect(None if rej is None else rej == "none", sc, "rej", "on: rejected: none (last interval)", rej or "")
    s_off, s_on = num(off.stat("surfaces")), num(on.stat("surfaces"))
    h_off, h_on = num(off.stat("hidden")), num(on.stat("hidden"))
    expect(both(s_off, s_on, lambda a, b: a == b and h_off == h_on), sc, "inst",
           "surfaces and hidden in the last [SceneStats] line, on == off", f"off {s_off}/{h_off}, on {s_on}/{h_on}")

    if sc == "repeat":
        moved = num(on.lastInputs("moved"))
        expect(None if moved is None else moved == 0, sc, "A1", "on: moved=0 (static props: a move is a handle swap)", f"moved={moved}")
        acc, rel = num(on.lastDirect("accepted")), num(on.lastDirect("released"))
        expect(both(acc, rel, lambda a, b: a == 0 and b == 0), sc, "D2",
               "on: accepted=0 released=0 (culled crates are anti-culled, not collected and built again)", f"accepted={acc} released={rel}")
        new = val(off.identity[-1], "new") if off.identity else None
        expect(None if new is None else new == 0, sc, "ref", "off: [D3d9Identity] new=0 (the old path keeps them)", f"new={new}")
    elif sc == "twice-overlay-first":
        # The crate is first drawn blended, so it is unordered and its six quads become billboards with split
        # opacity-micromap geometry, one surface each: the inst check (on == off) is the test.
        report("INFO", sc, "A4", "surfaces: the blended-first crate counts once per billboard quad", f"off {s_off}, on {s_on}")
    elif sc in ("twice", "twice-overlay"):
        expect(None if s_off is None else s_off == 2, sc, "ref", "off: 2 surfaces (ground and one crate instance)", f"{s_off}")
        expect(None if s_on is None else s_on == 2, sc, "A4", "on: 2 surfaces (the second draw merges into the first)", f"{s_on}")
    elif sc == "worldspace":
        report("INFO", sc, "A3", "on: per-frame updates / changed (both props are Updates every frame; a tie swap is not "
               "counted until the proposed swaps= counter exists)",
               f"updates={num(on.lastInputs('updates'))} changed={num(on.lastInputs('changed'))}")
    elif sc == "skinned":
        acc = num(on.lastDirect("accepted"))
        expect(None if acc is None else acc == 0, sc, "A1s", "on: accepted=0 in the interval with the bone switch (no new object)", f"accepted={acc}")
        report("INFO", sc, "E4", "on: skinned / updates per frame",
               f"skinned={num(on.lastInputs('skinned'))} updates={num(on.lastInputs('updates'))}")
        if off.identity:
            report("INFO", sc, "ref", "off: [D3d9Identity] new per frame", f"{val(off.identity[-1], 'new')}")
    elif sc == "identity-view":
        first_off = num(off.stat("surfaces", 0))
        expect(both(first_off, s_off, lambda a, b: a > b), sc, "ref",
               "off: the crates behind the camera are collected after the event (else the scenario tests nothing)",
               f"before {first_off}, after {s_off}")
        report("INFO", sc, "D1", "anti-culling must be off with an identity view: the surfaces check above is the test",
               f"off {s_off}, on {s_on}")
    elif sc in ("fused-world", "fused-view"):
        expect(None if ne is None else "inputs=" not in ne, sc, "E3", "on: no notEligible inputs class (fused draws handed over)", ne or "")
    elif sc == "reset":
        first_off, first_on = num(off.stat("surfaces", 0)), num(on.stat("surfaces", 0))
        expect(both(first_on, s_on, lambda a, b: a == b), sc, "M1", "on: surfaces after the Reset == before (no doubled objects left)",
               f"before {first_on}, after {s_on}")
        expect(both(first_off, s_off, lambda a, b: a == b), sc, "ref", "off: surfaces after == before", f"before {first_off}, after {s_off}")
        acc = num(on.lastDirect("accepted"))
        expect(None if acc is None else acc == 0, sc, "M1r",
               "on: accepted=0 in the Reset interval (recreated buffers re-associated, fix 5)", f"accepted={acc} released={num(on.lastDirect('released'))}")
    elif sc == "no-present":
        acc = num(on.lastDirect("accepted"))
        # Native window: the runtime's window procedure never passes these messages on (see the docs), so a pile of
        # unpresented draws fails this until that is fixed.
        expect(None if acc is None else acc == 0, sc, "M2", "on: accepted=0 after the unpresented stretch (no pile of draws)", f"accepted={acc}")
        new = val(off.identity[-1], "new") if off.identity else None
        expect(None if new is None else new == 0, sc, "ref", "off: [D3d9Identity] new=0", f"new={new}")
        for mode, log in (("off", off), ("on", on)):
            for l in log.out:
                if l.startswith("no-present:"):
                    report("INFO", sc, "M2", f"{mode}: {l[len('no-present: '):]}")
    elif sc == "sky":
        first = (num(off.stat("surfaces", 0)), num(off.stat("hidden", 0)))
        expect(None if s_off is None else first != (s_off, h_off), sc, "ref",
               "off: the reclassification changes surfaces or hidden (else the scenario tests nothing)", f"before {first}, after {(s_off, h_off)}")
        report("INFO", sc, "B1", "on: rasterOnly per frame (the sky service's replay of the crate)", f"rasterOnly={num(on.lastInputs('rasterOnly'))}")
    elif sc == "terrain":
        bakes, misses = num(on.lastDirect("bakes")), num(on.lastDirect("misses"))
        expect(both(bakes, misses, lambda a, b: a > 0 and b == 0), sc, "F3", "on: terrain bakes>0 misses=0", f"bakes={bakes} misses={misses}")
    elif sc == "vtf":
        cap = num(on.lastInputs("capture"))
        expect(None if cap is None else cap >= 1, sc, "ref", "on: the crate is vertex-captured (capture>=1)", f"capture={cap}")
        hits = num(on.lastInputs("hits"))
        expect(None if hits is None else hits == 0, sc, "B3", "on: early hits=0 (a vertex-texture draw is never skipped)", f"hits={hits}")
    elif sc == "portals":
        p_off, p_on = num(off.stat("portals")), num(on.stat("portals"))
        expect(None if p_off is None else p_off == 2, sc, "ref", "off: portals=2 (the pair is registered)", f"{p_off}")
        expect(None if p_on is None else p_on == 2, sc, "H3", "on: portals=2 (the pair is registered every frame)", f"{p_on}")
        report("INFO", sc, "H3", "portal instances hidden as duplicates of a portal index (total)",
               f"off {num(off.stat('portalDuplicatesHidden'))}, on {num(on.stat('portalDuplicatesHidden'))}")
    elif sc in ("emitter", "rendertarget"):
        ids = {"emitter": "G1", "rendertarget": "H6"}
        report("INFO", sc, ids[sc], "covered by the coverage and surfaces checks above")

print(f"D3D9V2 SCENARIOS DONE: {results['PASS']} pass, {results['FAIL']} fail")
PY
