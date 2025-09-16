#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_kpis.py
讀 output/kpis_*.csv，計算：
  1) success_rate           : 停車成功率（CSV 中有記錄的都算成功）
  2) avg_driving_distance_m : 平均行駛距離（controller 以 m/s*1s 累計）
  3) correct_reserved_rate  : family/disabled 是否停在對應保留位
輸出到 output/evaluation_summary.csv，並在終端印出表格。
"""
import glob
import csv
import os


def area_type(pa):
    if pa.startswith("PA_F_"):
        return "family"
    if pa.startswith("PA_D_"):
        return "disabled"
    return "general"


def read_kpis(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    n = len(rows)
    success_rate = 1.0 if n > 0 else 0.0
    dists = [float(r["driving_dist"])
             for r in rows if r.get("driving_dist", "")]
    avg_dist = sum(dists)/len(dists) if dists else 0.0
    ok = bad = 0
    for r in rows:
        u = r["userType"]
        pa = r["park_area"]
        t = area_type(pa)
        if u in ("family", "disabled"):
            if u == t:
                ok += 1
            else:
                bad += 1
    corr = ok / (ok+bad) if (ok+bad) > 0 else 0.0
    return {"n_parked": n, "success_rate": success_rate, "avg_driving_distance_m": avg_dist, "correct_reserved_rate": corr}


def main():
    files = sorted(glob.glob("output/kpis_*.csv"))
    if not files:
        print("No kpis_*.csv found under output/. Run a simulation first.")
        return
    rows = []
    for fp in files:
        sc = os.path.basename(fp)[5:-4]  # kpis_{scenario}.csv
        S = summarize(read_kpis(fp))
        S["scenario"] = sc
        rows.append(S)
        print(f"{sc:>14} | parked={S['n_parked']:4d} | success={S['success_rate']:.2f} | "
              f"avg_drive={S['avg_driving_distance_m']:.1f} | correct_reserved={S['correct_reserved_rate']:.2f}")
    out = "output/evaluation_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
                           "scenario", "n_parked", "success_rate", "avg_driving_distance_m", "correct_reserved_rate"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
