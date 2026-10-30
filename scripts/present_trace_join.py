#!/usr/bin/env python3
"""Line a PresentMon capture up against Remix's per-present trace (rtx.present.traceCsv).

Usage:
  presentmon --process_name Fallout4.exe --qpc_time_ms --timed 20 --terminate_after_timed --output_file pm.csv
  (with rtx.present.traceCsv = True in rtx.conf or the dev menu, so rtx-remix/logs/remix-present-trace.csv is written)
  python scripts/present_trace_join.py pm.csv "<game dir>/rtx-remix/logs/remix-present-trace.csv" [--dump N]

Both files carry QueryPerformanceCounter time in milliseconds (PresentMon: TimeInMs with --qpc_time_ms; Remix:
callQpcMs / gpuDoneQpcMs), so no offset has to be guessed. The NVIDIA present layer issues its DXGI Present calls
in the order it received vkQueuePresentKHR, one per Remix present, so the k-th PresentMon row matches the k-th
Remix present once the start of the capture is aligned. Alignment: pick the Remix row offset that minimises the
median (DXGI Present time - vkQueuePresentKHR time) over the first rows while keeping it non-negative.

Per matched present the script prints, for the stalls PresentMon saw (MsBetweenPresents > threshold):
  layerHoldMs     = DXGI Present call - vkQueuePresentKHR call        (time the layer's worker sat on this present)
  remixGpuDoneMs  = Remix GPU done   - vkQueuePresentKHR call        (the runtime's own view of when the frame finished)
  dxgiReadyMs     = MsRenderPresentLatency                            (DXGI/PresentMon: Present call -> frame ready to flip)
  displayedMs     = MsUntilDisplayed                                  (Present call -> on screen)
If remixGpuDoneMs stays ~1 frame while dxgiReadyMs / layerHoldMs jump to 50-90 ms on the stall rows, the stall is
between Remix's present semaphore and the flip, i.e. inside the layer / DXGI, not in Remix's submission or GPU work.
"""
import csv
import statistics
import sys


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def load_presentmon(path):
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        sys.exit("empty PresentMon csv")
    if "TimeInMs" not in rows[0]:
        sys.exit("PresentMon csv has no TimeInMs column (use PresentMon 2.x)")
    t0 = fnum(rows[0]["TimeInMs"])
    if t0 < 1e5:
        print("WARNING: TimeInMs starts at %.1f ms; this capture was probably made without --qpc_time_ms, "
              "so the times cannot be matched to the Remix trace." % t0)
    return rows


def load_trace(path):
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        sys.exit("empty Remix trace csv")
    for r in rows:
        for k in ("callQpcMs", "gpuDoneQpcMs", "gpuDoneAfterCallMs", "queuePresentMs", "gateWaitMs"):
            r[k] = fnum(r[k])
    return rows


def align(pm, tr, probe=60):
    """Return the index into tr that matches pm[0]."""
    pm_t = [fnum(r["TimeInMs"]) for r in pm[:probe]]
    best = None
    # candidate: the latest Remix present called no later than the first DXGI present, then slide back a few
    calls = [r["callQpcMs"] for r in tr]
    j0 = max(i for i, c in enumerate(calls) if c <= pm_t[0]) if any(c <= pm_t[0] for c in calls) else 0
    for j in range(max(0, j0 - 12), j0 + 1):
        holds = []
        ok = True
        for i, t in enumerate(pm_t):
            if j + i >= len(calls):
                ok = False
                break
            h = t - calls[j + i]
            if h < -0.5:
                ok = False
                break
            holds.append(h)
        if not ok or not holds:
            continue
        score = statistics.median(holds)
        if best is None or score < best[0]:
            best = (score, j)
    if best is None:
        sys.exit("could not align the two captures (do they overlap in time, and was --qpc_time_ms used?)")
    return best[1], best[0]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        sys.exit(__doc__)
    dump = 0
    if "--dump" in sys.argv:
        dump = int(sys.argv[sys.argv.index("--dump") + 1])
    thresh = 45.0
    pm = load_presentmon(args[0])
    tr = load_trace(args[1])
    j, med_hold = align(pm, tr)
    print("aligned: PresentMon row 0 = Remix present %s (trace row %d); median layer hold over the probe %.2f ms"
          % (tr[j]["present"], j, med_hold))

    n = min(len(pm), len(tr) - j)
    holds, remix_done, dxgi_ready, disp = [], [], [], []
    stall_rows = []
    for i in range(n):
        p, r = pm[i], tr[j + i]
        t_present = fnum(p["TimeInMs"])
        hold = t_present - r["callQpcMs"]
        rdone = r["gpuDoneQpcMs"] - r["callQpcMs"]
        ready = fnum(p.get("MsRenderPresentLatency", "nan"))
        shown = fnum(p.get("MsUntilDisplayed", "nan"))
        holds.append(hold)
        remix_done.append(rdone)
        dxgi_ready.append(ready)
        disp.append(shown)
        if fnum(p["MsBetweenPresents"]) > thresh:
            stall_rows.append(i)

    def pct(v, q):
        v = sorted(x for x in v if x == x)
        return v[min(len(v) - 1, int(len(v) * q))] if v else float("nan")

    print("matched %d presents; PresentMon stalls (MsBetweenPresents > %.0f ms): %d" % (n, thresh, len(stall_rows)))
    print("%-26s p50=%7.2f p95=%7.2f p99=%7.2f max=%7.2f" % ("layerHoldMs", pct(holds, .5), pct(holds, .95), pct(holds, .99), max(holds)))
    print("%-26s p50=%7.2f p95=%7.2f p99=%7.2f max=%7.2f" % ("remixGpuDoneMs", pct(remix_done, .5), pct(remix_done, .95), pct(remix_done, .99), max(remix_done)))
    print("%-26s p50=%7.2f p95=%7.2f p99=%7.2f max=%7.2f" % ("dxgiReadyMs", pct(dxgi_ready, .5), pct(dxgi_ready, .95), pct(dxgi_ready, .99), max(x for x in dxgi_ready if x == x)))
    print("%-26s p50=%7.2f p95=%7.2f p99=%7.2f max=%7.2f" % ("displayedMs", pct(disp, .5), pct(disp, .95), pct(disp, .99), max(x for x in disp if x == x)))

    # The frame that was on screen during a stall is the one presented just before the long MsBetweenPresents row.
    print("\nstall rows (the present BEFORE each gap is the frame that sat on screen):")
    print("%6s %6s %10s %10s %10s %10s %10s %6s" % ("pmRow", "remix", "betweenPr", "layerHold", "remixDone", "dxgiReady", "displayed", "fif"))
    for i in stall_rows[:40]:
        for k in (i - 1, i):
            if k < 0:
                continue
            p, r = pm[k], tr[j + k]
            print("%6d %6s %10.2f %10.2f %10.2f %10.2f %10.2f %6s%s" % (
                k, r["present"], fnum(p["MsBetweenPresents"]), holds[k], remix_done[k], dxgi_ready[k], disp[k],
                r["framesInFlight"], "  <- gap" if k == i else ""))
    if dump:
        print("\nfirst %d matched rows:" % dump)
        for k in range(min(dump, n)):
            p, r = pm[k], tr[j + k]
            print("%6d %6s %10.2f %10.2f %10.2f %10.2f %10.2f" % (
                k, r["present"], fnum(p["MsBetweenPresents"]), holds[k], remix_done[k], dxgi_ready[k], disp[k]))


if __name__ == "__main__":
    main()
