#!/usr/bin/env python
import os, sys, time, argparse, random, yaml, math, csv
from collections import defaultdict
import traci  # type: ignore

CFG_FILE = "config.sumocfg"

SPECIAL_FAMILY = ["PA_F_W"]
SPECIAL_DISABLED = ["PA_D_E"]
GENERAL = ["PA_G_W1", "PA_G_E1"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true", help="run with sumo-gui")
    ap.add_argument("--cfg", default=CFG_FILE)
    return ap.parse_args()

def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def sumo_cmd(gui: bool, cfg: str, params):
    exe = "sumo-gui" if gui else "sumo"
    cmd = [exe, "-c", cfg]
    if params.get("enable_emissions", False):
        os.makedirs("output", exist_ok=True)
        cmd += ["--emission-output", "output/emissions.xml"]
    return cmd

def kpis_header():
    return ["vehID","userType","scenario","t_enter","t_park","search_time",
            "driving_dist","reassigns","park_area","compliant","violation_flag"]

def write_kpi_row(w, row):
    w.writerow(row)

def user_type_from_vtype(vtype):
    if "family" in vtype: return "family"
    if "disabled" in vtype: return "disabled"
    return "normal"

def legal_candidates(userType):
    if userType == "family":
        return SPECIAL_FAMILY + GENERAL
    if userType == "disabled":
        return SPECIAL_DISABLED + GENERAL
    return GENERAL

def is_special(area_id):
    return area_id in SPECIAL_FAMILY or area_id in SPECIAL_DISABLED

def area_score(area_id, veh_id, params):
    # Simple heuristic score: ETA proxy + walking proxy + congestion penalty
    occ = traci.parkingarea.getVehicleCount(area_id)
    cap = traci.parkingarea.getRoadSideCapacity(area_id)
    occ_ratio = (occ+1) / max(1, cap)
    base_drive = random.uniform(5, 20)  # placeholder
    walk_proxy = 30 if is_special(area_id) else 40  # assume special nearer
    cong = 20 * occ_ratio
    return base_drive + 0.3*walk_proxy + cong

class Controller:
    def __init__(self, params):
        self.p = params
        self.scenario = params.get("scenario", "assign_dynamic")
        random.seed(params.get("random_seed", 42))
        self.users = {}  # vehID -> dict
        os.makedirs("output", exist_ok=True)
        self.kpi_file = open("output/kpis.csv", "w", newline="", encoding="utf-8")
        self.kpi_writer = csv.writer(self.kpi_file)
        self.kpi_writer.writerow(kpis_header())
        self.last_enforce = 0

    def on_depart(self, veh_id):
        vtype = traci.vehicle.getTypeID(veh_id)
        ut = user_type_from_vtype(vtype)
        self.users[veh_id] = {
            "userType": ut,
            "assigned": None,
            "reassigns": 0,
            "t_enter": traci.simulation.getTime(),
            "driving_dist": 0.0,
            "compliant": True,
            "violation_flag": False,
        }
        if self.scenario == "free":
            return  # no assignment
        # compliance
        if random.random() > self.p.get("p_compliance", 1.0):
            self.users[veh_id]["compliant"] = False
            return
        # initial assignment
        cand = legal_candidates(ut)
        best = min(cand, key=lambda a: area_score(a, veh_id, self.p))
        self.assign_to_area(veh_id, best)

    def assign_to_area(self, veh_id, area_id):
        # respect capacity; if full, do nothing (reassignment tick will handle)
        occ = traci.parkingarea.getVehicleCount(area_id)
        cap = traci.parkingarea.getRoadSideCapacity(area_id)
        if occ >= cap:
            return
        try:
            traci.vehicle.setParkingAreaStop(veh_id, area_id, duration=random.randint(300,900))
            self.users[veh_id]["assigned"] = area_id
        except traci.TraCIException:
            pass

    def tick(self, t):
        # track distance (m/s * 1s)
        for v in traci.vehicle.getIDList():
            try:
                self.users[v]["driving_dist"] += traci.vehicle.getSpeed(v)
            except KeyError:
                pass
        # dynamic reassignment
        if self.scenario == "assign_dynamic":
            for v in list(traci.vehicle.getIDList()):
                u = self.users.get(v, {})
                tgt = u.get("assigned")
                if tgt is None: 
                    continue
                if u.get("reassigns",0) < self.p.get("max_reassignments",1):
                    occ = traci.parkingarea.getVehicleCount(tgt)
                    cap = traci.parkingarea.getRoadSideCapacity(tgt)
                    if occ >= cap:
                        ut = u.get("userType","normal")
                        cand = [c for c in legal_candidates(ut) if c != tgt]
                        if cand:
                            best = min(cand, key=lambda a: area_score(a, v, self.p))
                            self.assign_to_area(v, best)
                            self.users[v]["reassigns"] += 1
        # enforcement
        if self.p.get("enable_enforcement", False):
            per = self.p.get("enforcement_period_s", 30)
            if t - self.last_enforce >= per:
                self.last_enforce = t
                self.enforcement_sweep()

    def enforcement_sweep(self):
        p_detect = self.p.get("p_detect", 0.7)
        for pa in SPECIAL_FAMILY + SPECIAL_DISABLED:
            parked = traci.parkingarea.getVehicleIDs(pa)
            for v in parked:
                u = self.users.get(v, {})
                if not u: 
                    continue
                if u["userType"] == "normal" and random.random() < p_detect:
                    self.users[v]["violation_flag"] = True
                    try:
                        traci.vehicle.resume(v)
                    except traci.TraCIException:
                        pass

    def on_park(self, veh_id, area_id):
        u = self.users.get(veh_id, {})
        t = traci.simulation.getTime()
        if not u:
            return
        write_kpi_row(self.kpi_writer, [
            veh_id, u["userType"], self.scenario, u["t_enter"], t,
            t - u["t_enter"], round(u["driving_dist"],1), u["reassigns"],
            area_id, int(u["compliant"]), int(u["violation_flag"])
        ])
        self.kpi_file.flush()

def main():
    args = parse_args()
    params = load_params()
    cmd = sumo_cmd(args.gui, args.cfg, params)
    traci.start(cmd)
    ctl = Controller(params)

    seen = set()
    parked_prev = set()
    try:
        # Step once BEFORE checking minExpected to avoid 0.00 termination
        traci.simulationStep()
        while True:
            t = int(traci.simulation.getTime())

            # vehicles that departed this step
            for v in traci.simulation.getDepartedIDList():
                if v not in seen:
                    seen.add(v)
                    ctl.on_depart(v)
            # (safety) handle late-seen vehicles
            for v in traci.vehicle.getIDList():
                if v not in seen:
                    seen.add(v)
                    ctl.on_depart(v)

            # detect parking events
            parked_now = set()
            for pa in GENERAL + SPECIAL_FAMILY + SPECIAL_DISABLED:
                for v in traci.parkingarea.getVehicleIDs(pa):
                    parked_now.add((v, pa))
            newly_parked = parked_now - set(parked_prev)
            for v, pa in newly_parked:
                ctl.on_park(v, pa)
            parked_prev = parked_now  # <- update history

            ctl.tick(t)

            # exit conditions
            if traci.simulation.getMinExpectedNumber() == 0 and traci.simulation.getTime() >= 1.0:
                break

            traci.simulationStep()
    finally:
        traci.close()

if __name__ == "__main__":
    main()
