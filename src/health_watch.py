"""Watch a training run and the Mac's health. Prints one line per EVENT (for Claude's Monitor):
  - each validation result (every --steps-per-eval steps) with temps / memory since the last one
  - health alerts, printed once when a condition starts and once when it clears
  - DONE / CRASHED when the training process ends
Every 60 s it also appends a row to health.log (CSV) so the whole night can be reviewed later.

Protective action: if macOS reports a CRITICAL thermal state, any temperature reaches 105 C,
or memory pressure is critical with heavy swap, the training process is PAUSED (SIGSTOP), never killed.
Resume with:  kill -CONT <pid>   (the pid is printed in the alert)

  /usr/bin/python3 -I src/health_watch.py lfm3_run1.log
Safe to restart at any time: progress is kept in .health_state.json.
"""
import ctypes
import ctypes.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "lfm3_run1.log")
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 9455
STATE = ROOT / f".health_state_{LOG.stem}.json"
PATTERN = sys.argv[3] if len(sys.argv) > 3 else "bin/python src/train.py"
CSV = ROOT / "health.log"
ANSI = re.compile(r"\x1b\[[0-9;]*m")

WARN_TEMP, PAUSE_TEMP = 95.0, 105.0      # Apple Silicon throttles itself around 100-110 C
WARN_SWAP_GB, PAUSE_SWAP_GB = 3.0, 8.0


def say(msg):
    print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)


def thermal_state():
    """macOS's own thermal verdict: nominal / fair / serious / critical."""
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        pi = objc.objc_msgSend(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
        objc.objc_msgSend.restype = ctypes.c_long
        return ["nominal", "fair", "serious", "critical"][objc.objc_msgSend(pi, objc.sel_registerName(b"thermalState"))]
    except Exception:
        return "unknown"


def sample():
    s = {"thermal": thermal_state()}
    try:
        out = subprocess.run(["macmon", "pipe", "-s", "1", "-i", "1000"], capture_output=True, text=True, timeout=20).stdout
        d = json.loads(out.strip().splitlines()[-1])
        s.update(cpu_c=d["temp"]["cpu_temp_avg"], gpu_c=d["temp"]["gpu_temp_avg"],
                 gpu_busy=d.get("gpu_active_ratio", 0), watts=d.get("all_power", 0),
                 mem_pressure=d["memory"]["pressure_level"],
                 ram_gb=d["memory"]["ram_usage"] / 1e9, swap_gb=d["memory"]["swap_usage"] / 1e9)
    except Exception as e:
        s["macmon_error"] = str(e)[:80]
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    s["on_ac"] = "AC Power" in batt
    m = re.search(r"(\d+)%", batt)
    s["battery"] = int(m.group(1)) if m else None
    return s


def train_pid():
    out = subprocess.run(["pgrep", "-f", PATTERN], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def parse_log():
    lines = ANSI.sub("", LOG.read_text(errors="ignore")).splitlines() if LOG.exists() else []
    vals = [(int(m.group(1)), float(m.group(2))) for l in lines if (m := re.match(r"\s*(\d+)\s+val\s+([\d.]+)", l))]
    trains = [(int(m.group(1)), float(m.group(2))) for l in lines if (m := re.match(r"\s*(\d+)\s+([\d.]+)\s", l))]
    return lines, vals, trains


def main():
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    st.setdefault("reported", [])
    st.setdefault("alerts", {})
    st.setdefault("window", {"max_cpu": 0, "max_gpu": 0, "max_swap": 0, "worst_thermal": "nominal", "since": time.time()})
    if not CSV.exists():
        CSV.write_text("time,step,thermal,cpu_c,gpu_c,gpu_busy,watts,ram_gb,swap_gb,mem_pressure,on_ac,battery\n")
    order = ["nominal", "fair", "serious", "critical", "unknown"]
    first_ts = None
    last_csv = 0
    say(f"watching {LOG.name}; reported so far: {st['reported'] or 'none'}")

    while True:
        lines, vals, trains = parse_log()
        step = trains[-1][0] if trains else 0
        s = sample()
        w = st["window"]
        if "cpu_c" in s:
            w["max_cpu"] = max(w["max_cpu"], s["cpu_c"])
            w["max_gpu"] = max(w["max_gpu"], s["gpu_c"])
            w["max_swap"] = max(w["max_swap"], s["swap_gb"])
        if order.index(s["thermal"]) > order.index(w["worst_thermal"]) and s["thermal"] != "unknown":
            w["worst_thermal"] = s["thermal"]

        if time.time() - last_csv >= 60:
            last_csv = time.time()
            with CSV.open("a") as f:
                f.write(",".join(str(x) for x in [time.strftime("%H:%M:%S"), step, s["thermal"],
                        *(round(s.get(k, -1), 2) for k in ["cpu_c", "gpu_c", "gpu_busy", "watts", "ram_gb", "swap_gb"]),
                        s.get("mem_pressure", -1), s["on_ac"], s["battery"]]) + "\n")

        # ---- new validation results
        for it, v in vals:
            if it in st["reported"] or it == 1:
                continue
            prev = [x for x in vals if x[0] < it]
            seg = [l for i, l in trains if it - 500 < i <= it]
            first_ts = first_ts or os.path.getmtime(LOG)
            say(f"VAL step {it}/{TOTAL}: val {v:.3f} (prev {prev[-1][1]:.3f}, start {vals[0][1]:.3f}) | "
                f"train avg last 500 {sum(seg) / max(len(seg), 1):.3f} | "
                f"since last report: max CPU {w['max_cpu']:.0f}C, max GPU {w['max_gpu']:.0f}C, "
                f"worst thermal {w['worst_thermal']}, max swap {w['max_swap']:.1f}GB | "
                f"now: thermal {s['thermal']}, RAM {s.get('ram_gb', -1):.1f}GB, AC {s['on_ac']}")
            st["reported"].append(it)
            st["window"] = {"max_cpu": 0, "max_gpu": 0, "max_swap": 0, "worst_thermal": "nominal", "since": time.time()}

        # ---- health alerts (printed on start and on clear)
        pid = train_pid()
        hot = max(s.get("cpu_c", 0), s.get("gpu_c", 0))
        conds = {
            "thermal_serious": s["thermal"] in ("serious", "critical"),
            "temp_high": hot >= WARN_TEMP,
            "swap_high": s.get("swap_gb", 0) >= WARN_SWAP_GB,
            "mem_pressure": s.get("mem_pressure", 0) >= 1,
            "on_battery": not s["on_ac"],
            "macmon_failed": "macmon_error" in s,
        }
        for k, on in conds.items():
            if on and not st["alerts"].get(k):
                say(f"ALERT {k}: thermal {s['thermal']}, CPU {s.get('cpu_c', -1):.0f}C, GPU {s.get('gpu_c', -1):.0f}C, "
                    f"swap {s.get('swap_gb', -1):.1f}GB, mem_pressure {s.get('mem_pressure')}, AC {s['on_ac']} "
                    f"battery {s['battery']}% {s.get('macmon_error', '')}")
            elif not on and st["alerts"].get(k):
                say(f"CLEARED {k}")
            st["alerts"][k] = on

        danger = (s["thermal"] == "critical" or hot >= PAUSE_TEMP
                  or (s.get("mem_pressure", 0) >= 2 and s.get("swap_gb", 0) >= PAUSE_SWAP_GB)
                  or (not s["on_ac"] and (s["battery"] or 100) <= 20))
        if danger and pid and not st.get("paused"):
            os.kill(pid, signal.SIGSTOP)
            st["paused"] = pid
            say(f"PAUSED training (pid {pid}) to protect the Mac: thermal {s['thermal']}, max temp {hot:.0f}C, "
                f"swap {s.get('swap_gb', -1):.1f}GB, AC {s['on_ac']}. Resume: kill -CONT {pid}")

        # ---- end of run
        text = "\n".join(lines[-40:])
        if re.search(r"Traceback|Error|Killed|Abort", text) and not st.get("crash_reported"):
            st["crash_reported"] = True
            say("CRASHED? log shows: " + " | ".join(l for l in lines[-6:] if l.strip()))
        if pid is None:
            time.sleep(10)  # let the pipe (grep | tee) flush the final 'done in' line
            lines, vals, trains = parse_log()
            STATE.write_text(json.dumps(st))
            done = [l for l in lines if l.startswith("done")]
            say(("DONE " + done[-1]) if done else "STOPPED: training process is gone without a 'done' line. Last: "
                + " | ".join(l for l in lines[-4:] if l.strip()))
            return
        STATE.write_text(json.dumps(st))
        time.sleep(30)


if __name__ == "__main__":
    main()
