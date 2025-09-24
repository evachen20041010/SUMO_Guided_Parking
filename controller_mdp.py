#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
controller_mdp.py  (legacy TraCI-friendly)
- 不使用 traci.parkingarea.getCapacity()
- 會從 config.sumocfg 解析 additional-files，讀 parking.add.xml 取得各 PA 容量
- 佔用數用 getVehicleCount()；若舊版無此函式則改用 len(getVehicleIDs())
- 其它邏輯同前：MDP/啟發式/動態重指派、KPI、run.log
"""
import os
import sys
import argparse
import random
import yaml
import math
import csv
import traceback
from collections import defaultdict
import xml.etree.ElementTree as ET
import traci  # pip install eclipse-sumo

CFG_FILE = "config.sumocfg"
LOG_PATH = "output/run.log"

# === 依你的車位 ID 命名分群（若改名請同步改這邊） ===
SPECIAL_FAMILY = ["PA_F_W"]
SPECIAL_DISABLED = ["PA_D_E"]
GENERAL = ["PA_G_W1", "PA_G_E1"]
ALL_PAS = GENERAL + SPECIAL_FAMILY + SPECIAL_DISABLED

# ---- 全域：由 add.xml 解析出的容量快取 ----
CAP_CACHE = {}  # pa_id -> int(capacity)

# ---------- 小工具 ----------


def pa_type(pa_id: str) -> str:
    if pa_id in SPECIAL_FAMILY:
        return "family"
    if pa_id in SPECIAL_DISABLED:
        return "disabled"
    return "general"


def user_type_from_vtype(vtype: str) -> str:
    vt = vtype.lower()
    if "family" in vt:
        return "family"
    if "disabled" in vt:
        return "disabled"
    if "vip" in vt:
        return "vip"
    return "normal"


def legal_candidates(user_type: str):
    if user_type == "family":
        return SPECIAL_FAMILY + GENERAL
    if user_type == "disabled":
        return SPECIAL_DISABLED + GENERAL
    if user_type == "vip":
        return GENERAL
    return GENERAL

# --- 佔用/容量（相容舊版 TraCI） ---


def pa_occupancy(pa_id: str) -> int:
    """優先用 getVehicleCount；沒有就用 getVehicleIDs"""
    try:
        return int(traci.parkingarea.getVehicleCount(pa_id))
    except Exception:
        try:
            return len(traci.parkingarea.getVehicleIDs(pa_id))
        except Exception:
            return 0


def pa_capacity(pa_id: str, log=None) -> int:
    """優先用 CAP_CACHE；若沒有，就回推 999（避免誤判滿位）。"""
    cap = CAP_CACHE.get(pa_id)
    if cap is not None:
        return cap
    if log:
        log(f"[warn] capacity not found in CAP_CACHE for {pa_id}. Assume 999.")
    return 999

# ---------- 參數 ----------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true", help="run with sumo-gui")
    ap.add_argument("--cfg", default=CFG_FILE)
    return ap.parse_args()


def load_params():
    try:
        with open("params.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def sumo_cmd(gui: bool, cfg: str, params):
    exe = "sumo-gui" if gui else "sumo"
    cmd = [exe, "-c", cfg]
    if params.get("enable_emissions", False):
        os.makedirs("output", exist_ok=True)
        cmd += ["--emission-output", "output/emissions.xml"]
    return cmd

# ---------- 讀 sumocfg / parking.add.xml 取得容量 ----------


def find_additional_files_from_sumocfg(cfg_path: str):
    """回傳 additional-files 的清單（展開逗號/分號）"""
    try:
        tree = ET.parse(cfg_path)
        root = tree.getroot()
        node = root.find(".//configuration/input/additional-files")
        if node is None:
            return []
        val = node.get("value", "")
        parts = [p.strip() for p in val.replace(
            ";", ",").split(",") if p.strip()]
        return parts
    except Exception:
        return []


def load_pa_capacities_from_additional(paths, log=None):
    """從 parking.add.xml 讀取 <parkingArea> 的 capacity
       支援屬性：roadsideCapacity / capacity；或計算 <space> 子節點數量"""
    caps = {}
    for p in paths:
        if not os.path.exists(p):
            # 有些 sumocfg 路徑是相對 config 的；試著用 cfg 所在目錄拼
            continue
        try:
            tree = ET.parse(p)
            root = tree.getroot()
            for pa in root.findall(".//parkingArea"):
                pid = pa.get("id")
                cap = pa.get("roadsideCapacity")
                if cap is None:
                    cap = pa.get("capacity")
                if cap is None:
                    # 有些寫成逐格 <space/>：數一下
                    cap = len(pa.findall(".//space"))
                if pid and cap:
                    try:
                        caps[pid] = int(float(cap))
                    except:
                        pass
        except Exception as e:
            if log:
                log(f"[warn] parse {p} failed: {e}")
    if log:
        log(f"[cap] loaded: {caps}")
    return caps

# ---------- 紀錄 ----------


def logger():
    os.makedirs("output", exist_ok=True)
    f = open(LOG_PATH, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        print(msg, file=f, flush=True)
    return log, f


# ---------- KPI ----------
KPI_HEADER = [
    "vehID", "userType", "scenario", "t_enter", "t_park", "search_time",
    "driving_dist", "reassigns", "park_area", "compliant", "violation_flag",
    "drive_cost", "walk_cost", "penalty_cost", "benefit", "var_term", "R_selected"
]


def open_kpi_writer(scenario: str):
    os.makedirs("output", exist_ok=True)
    path = f"output/kpis_{scenario}.csv"
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(KPI_HEADER)
    return f, w, path

# ---------- 幾何 & 路徑 ----------


def get_edge_of_pa(pa_id: str) -> str:
    lane_id = traci.parkingarea.getLaneID(pa_id)
    return traci.lane.getEdgeID(lane_id)


def route_length_time(from_edge: str, to_edge: str):
    try:
        r = traci.simulation.findRoute(from_edge, to_edge, vType="car_normal")
        return r.length, r.travelTime
    except traci.TraCIException:
        return float("inf"), float("inf")

# ---------- MDP 成本/獎勵 ----------


def driving_cost(v_id: str, pa_id: str) -> float:
    edge_now = traci.vehicle.getRoadID(v_id)
    edge_pa = get_edge_of_pa(pa_id)
    L, T = route_length_time(edge_now, edge_pa)
    return T if math.isfinite(T) else (L / 10.0)


def walking_cost(pa_id: str, mdp_cfg: dict) -> float:
    wd = mdp_cfg.get("walking_proxy", {}).get(pa_id)
    return float(wd) if wd is not None else (30.0 if pa_type(pa_id) in ("family", "disabled") else 40.0)


def violation_penalty(user_type: str, pa_t: str, mdp_cfg: dict) -> float:
    M = mdp_cfg.get("penalty_matrix", {
        "normal":   {"general": 0, "family": 3, "disabled": 3},
        "family":   {"general": 1, "family": 0, "disabled": 2},
        "disabled": {"general": 2, "family": 2, "disabled": 0},
        "vip":      {"general": 1, "family": 3, "disabled": 3},
    })
    base = M.get(user_type, M["normal"]).get(pa_t, 3)
    gamma = float(mdp_cfg.get("gamma_violation", 10.0))
    return gamma * base


def benefit(pa_id: str, user_type: str, mdp_cfg: dict) -> float:
    baseFee = float(mdp_cfg.get("base_fee", 0.0))
    zone_heat = mdp_cfg.get("zone_heat", {})
    user_bonus = mdp_cfg.get(
        "user_bonus", {"vip": 20, "disabled": 10, "family": 10, "normal": 0})
    return baseFee + float(zone_heat.get(pa_id, 0)) + float(user_bonus.get(user_type, 0))


def reserved_cost(pa_t: str, user_type: str, mdp_cfg: dict) -> float:
    m = float(mdp_cfg.get("m_reserved_mgmt", 5.0))
    M = float(mdp_cfg.get("M_violate", 50.0))
    if pa_t in ("family", "disabled"):
        return 0.0 if user_type == pa_t else M
    return m if user_type in ("family", "disabled", "vip") else 0.0


def variance_term(mdp_cfg: dict) -> float:
    """以各類剩餘量的方差做均衡懲罰（用 CAP_CACHE 與 pa_occupancy）"""
    rem = defaultdict(int)
    for pa in ALL_PAS:
        cap = pa_capacity(pa)
        occ = pa_occupancy(pa)
        rem[pa_type(pa)] += max(0, cap - occ)
    vals = list(rem.values())
    if not vals:
        return 0.0
    mu = sum(vals)/len(vals)
    var = sum((x-mu)**2 for x in vals)/len(vals)
    lam = float(mdp_cfg.get("lambda_var", 1.0))
    return lam * var


def mdp_reward(v_id: str, pa_id: str, user_type: str, mdp_cfg: dict):
    alpha = float(mdp_cfg.get("alpha_drive", 0.3))
    beta = float(mdp_cfg.get("beta_walk", 0.6))
    pa_t = pa_type(pa_id)
    d = driving_cost(v_id, pa_id)
    w = walking_cost(pa_id, mdp_cfg)
    p = violation_penalty(user_type, pa_t, mdp_cfg)
    b = benefit(pa_id, user_type, mdp_cfg)
    var = variance_term(mdp_cfg)
    R = b - (alpha*d + beta*w) - reserved_cost(pa_t, user_type, mdp_cfg) - var
    return R, (d, w, p, b, var)

# ---------- 控制器 ----------


class Controller:
    def __init__(self, params, log):
        self.p = params
        self.log = log
        self.scenario = params.get("scenario", "mdp")
        random.seed(params.get("random_seed", 42))
        self.users = {}  # vehID -> 狀態
        self.kfile, self.kw, self.kpath = open_kpi_writer(self.scenario)
        self.replan_dt = int(params.get("mdp_replan_period_s", 10))
        self.max_reassign = int(params.get("max_reassignments", 2))
        self.last_replan_at = 0

    def close(self):
        try:
            self.kfile.close()
        except:
            pass

    # --- 實際指派（改道 + 下停靠） ---
    def _reroute_to_pa(self, veh_id: str, pa_id: str):
        edge_id = get_edge_of_pa(pa_id)
        try:
            traci.vehicle.changeTarget(veh_id, edge_id)
        except traci.TraCIException:
            pass
        try:
            traci.vehicle.rerouteTraveltime(veh_id)
        except traci.TraCIException:
            pass
        try:
            traci.vehicle.setParkingAreaStop(
                veh_id, pa_id, duration=self.p.get("park_duration_s", 600)
            )
        except traci.TraCIException as e:
            self.log(f"[assign-failed] {veh_id} -> {pa_id}: {e}")

    def _select_candidates(self, user_type: str):
        allow_violation = bool(self.p.get("allow_violation", True))
        cands = list(ALL_PAS) if allow_violation else list(
            legal_candidates(user_type))
        # 過濾滿位（用 CAP_CACHE 容量 + 佔用）
        free = []
        for pa in cands:
            cap, occ = pa_capacity(pa), pa_occupancy(pa)
            if occ < cap:
                free.append(pa)
        return free

    # --- 事件：出發 ---
    def on_depart(self, veh_id: str):
        vtype = traci.vehicle.getTypeID(veh_id)
        ut = user_type_from_vtype(vtype)
        self.users[veh_id] = {
            "userType": ut, "assigned": None, "reassigns": 0,
            "t_enter": traci.simulation.getTime(), "driving_dist": 0.0,
            "compliant": True, "violation_flag": False,
            "mdp_terms": (None, None, None, None, None), "R": None
        }
        if self.scenario == "free":
            return
        if random.random() > self.p.get("p_compliance", 1.0):
            self.users[veh_id]["compliant"] = False
            return
        self._decide_and_assign(veh_id)

    def _decide_and_assign(self, veh_id: str):
        ut = self.users[veh_id]["userType"]
        if self.scenario == "mdp":
            mdp_cfg = self.p.get("mdp", {})
            cands = self._select_candidates(ut)
            if not cands:
                self.log(f"[mdp] no candidates for {veh_id} ({ut})")
                return
            best_pa, best_R, best_terms = None, -1e18, None
            for pa in cands:
                R, terms = mdp_reward(veh_id, pa, ut, mdp_cfg)
                if R > best_R:
                    best_R, best_pa, best_terms = R, pa, terms
            self.users[veh_id]["assigned"] = best_pa
            self.users[veh_id]["R"] = best_R
            self.users[veh_id]["mdp_terms"] = best_terms
            self._reroute_to_pa(veh_id, best_pa)
            self.log(
                f"[mdp] assign {veh_id} ({ut}) -> {best_pa}  R={best_R:.2f}")
        else:
            # 簡單啟發式（佔用率 + 步行代價）
            def h(pa):
                occ = pa_occupancy(pa)
                cap = pa_capacity(pa)
                occ_ratio = (occ+1) / max(1, cap)
                walk = 30.0 if pa_type(pa) in ("family", "disabled") else 40.0
                return 0.5*occ_ratio + 0.5*(walk/50.0)
            cands = self._select_candidates(ut)
            if not cands:
                return
            best = min(cands, key=h)
            self.users[veh_id]["assigned"] = best
            self._reroute_to_pa(veh_id, best)
            self.log(f"[heuristic] assign {veh_id} ({ut}) -> {best}")

    # --- 事件：實際停進去 ---
    def on_park(self, veh_id: str, area_id: str):
        u = self.users.get(veh_id, {})
        t = traci.simulation.getTime()
        if not u:
            return
        row = [
            veh_id, u["userType"], self.scenario,
            u["t_enter"], t, t - u["t_enter"],
            round(u["driving_dist"], 1), u["reassigns"],
            area_id, int(u["compliant"]), int(u["violation_flag"])
        ]
        if self.scenario == "mdp" and u["mdp_terms"][0] is not None:
            d, w, p, b, var = u["mdp_terms"]
            row += [round(d, 2), round(w, 2), round(p, 2),
                    round(b, 2), round(var, 2), round(u["R"], 2)]
        else:
            row += ["", "", "", "", "", ""]
        self.kw.writerow(row)

    # --- 每步更新 ---
    def tick(self, t_now: int):
        # 行駛距離（m/s * 1s）
        for v in traci.vehicle.getIDList():
            try:
                self.users[v]["driving_dist"] += traci.vehicle.getSpeed(v)
            except KeyError:
                pass

        # 動態重指派：目標滿了就重算
        if self.scenario in ("assign_dynamic", "mdp"):
            if t_now - self.last_replan_at >= self.replan_dt:
                self.last_replan_at = t_now
                for v in list(traci.vehicle.getIDList()):
                    u = self.users.get(v)
                    if not u or u["reassigns"] >= self.max_reassign:
                        continue
                    tgt = u.get("assigned")
                    if tgt is None:
                        continue
                    cap = pa_capacity(tgt)
                    occ = pa_occupancy(tgt)
                    if occ >= cap:  # 目標滿了 → 重新決策
                        self._decide_and_assign(v)
                        u["reassigns"] += 1

# ---------- 主迴圈 ----------


def main():
    os.makedirs("output", exist_ok=True)
    log, log_file = logger()

    try:
        args = parse_args()
        params = load_params()

        # 先從 sumocfg 找 additional-files，載入 capacity
        add_paths = find_additional_files_from_sumocfg(args.cfg)
        # 若 sumocfg 沒寫，嘗試用預設檔名
        if not add_paths and os.path.exists("parking.add.xml"):
            add_paths = ["parking.add.xml"]
        global CAP_CACHE
        CAP_CACHE = load_pa_capacities_from_additional(add_paths, log)

        cmd = sumo_cmd(args.gui, args.cfg, params)
        log(f"[launch] {' '.join(cmd)}")

        traci.start(cmd)
        log("[traci] connected")

        ctl = Controller(params, log)
        seen = set()
        parked_prev = set()

        traci.simulationStep()
        log(f"[step0] t={traci.simulation.getTime():.2f} minExpected={traci.simulation.getMinExpectedNumber()}")

        while True:
            t = int(traci.simulation.getTime())

            for v in traci.simulation.getDepartedIDList():
                if v not in seen:
                    seen.add(v)
                    ctl.on_depart(v)
            for v in traci.vehicle.getIDList():
                if v not in seen:
                    seen.add(v)
                    ctl.on_depart(v)

            parked_now = set()
            for pa in ALL_PAS:
                for v in traci.parkingarea.getVehicleIDs(pa):
                    parked_now.add((v, pa))
            newly = parked_now - set(parked_prev)
            for v, pa in newly:
                ctl.on_park(v, pa)
            parked_prev = parked_now

            ctl.tick(t)

            if traci.simulation.getTime().is_integer():
                me = traci.simulation.getMinExpectedNumber()
                log(f"[t={t:4d}] minExpected={me} running={len(traci.vehicle.getIDList())}")

            if traci.simulation.getMinExpectedNumber() == 0 and traci.simulation.getTime() >= 1.0:
                log("[exit] minExpected==0")
                break

            traci.simulationStep()

    except Exception:
        log("[ERROR]\n" + traceback.format_exc())
    finally:
        try:
            ctl.close()
        except:
            pass
        try:
            traci.close(False)
        except:
            pass
        try:
            log_file.close()
        except:
            pass


if __name__ == "__main__":
    main()
