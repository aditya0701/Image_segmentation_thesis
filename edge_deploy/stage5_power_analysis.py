"""
Stage 5: energy per inference from the HWiNFO sensor log.

For each 60 s sustained-load window (see stage5_openvino.py bench) and the two
60 s idle windows, average the HWiNFO power sensors over the window, then:
    energy per inference (J) = average power (W) * window length (s) / inferences
    above-idle energy (J)    = (average power - idle power) * window length / inferences
CPU Package Power covers the whole SoC package (CPU cores, Arc GPU, NPU, memory
controller) on this chip, so it is the main number. It is a software sensor
reading, not a power meter, and HWiNFO logged every 2 s.

Usage:
    D:/my_envs/sam-edge/python.exe edge_deploy/stage5_power_analysis.py
"""
import csv
import json
from datetime import datetime

import pandas as pd

import common
from stage5_openvino import HWINFO_LOG, POWER_WINDOWS

SENSORS = ["CPU Package Power [W]", "IA Cores Power [W]", "IGPU Power [W]", "Total System Power [W]"]


# HWiNFO writes the header once. Sensors that appear after logging starts (seen:
# battery "Estimated Remaining Time", "iGPU VID", "NPU Clock") are added to the
# rows but not the header. In both logs from this laptop they sit after the
# "Wear Level [%]" battery column, i.e. after every power sensor used here, so the
# columns we read keep their positions. Rows are only accepted if that holds.
STABLE_UNTIL = "Wear Level [%]"


def load_log():
    with open(HWINFO_LOG, encoding="cp1252", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    limit = header.index(STABLE_UNTIL)
    needed = [header.index(s) for s in SENSORS]
    assert max(needed) < limit, "a power sensor sits after the column where new sensors can appear"
    keep = limit + 1
    df = pd.DataFrame([(r + [""] * keep)[:keep] for r in rows[1:]], columns=header[:keep])
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")
    out = pd.DataFrame({s: pd.to_numeric(df[s], errors="coerce") for s in SENSORS})
    out["t"] = ts
    return out.dropna(subset=["t"])


def window_mean(log, start, end):
    sel = log[(log["t"] >= datetime.fromtimestamp(start)) & (log["t"] <= datetime.fromtimestamp(end))]
    return sel[SENSORS].mean(), len(sel)


def main():
    log = load_log()
    windows = json.loads(POWER_WINDOWS.read_text())
    idle = {}
    for key in ("idle_start", "idle_end"):
        idle[key], n = window_mean(log, windows[key]["start"], windows[key]["end"])
        print(f"{key}: package {idle[key]['CPU Package Power [W]']:.2f} W, system "
              f"{idle[key]['Total System Power [W]']:.2f} W ({n} samples)")
    idle_pkg = (idle["idle_start"]["CPU Package Power [W]"] + idle["idle_end"]["CPU Package Power [W]"]) / 2

    rows = []
    for key, w in windows.items():
        if key.startswith("idle") or key.endswith("_bench"):
            continue
        device, variant = key.split("_", 1)
        mean, n = window_mean(log, w["start"], w["end"])
        seconds = w["end"] - w["start"]
        per_inf_s = seconds / w["runs"]
        pkg = mean["CPU Package Power [W]"]
        rows.append({
            "device": device, "variant": variant, "inferences": w["runs"], "samples": n,
            "ms_per_inference": round(per_inf_s * 1000),
            "package_W": round(pkg, 2), "ia_cores_W": round(mean["IA Cores Power [W]"], 2),
            "igpu_W": round(mean["IGPU Power [W]"], 2), "system_W": round(mean["Total System Power [W]"], 2),
            "J_per_inference": round(pkg * per_inf_s, 2),
            "J_above_idle": round((pkg - idle_pkg) * per_inf_s, 2),
        })
    table = pd.DataFrame(rows).sort_values("J_per_inference")
    print(f"\nIdle package power (mean of both idle windows): {idle_pkg:.2f} W\n")
    print(table.to_string(index=False))
    table.to_csv(common.RESULTS_DIR / "stage5_energy.csv", index=False)


if __name__ == "__main__":
    main()
