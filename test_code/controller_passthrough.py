#!/usr/bin/env python
import argparse, sys, os
import traci

def build_cmd(gui: bool, cfg: str):
    sumo = "sumo-gui" if gui else "sumo"
    return [sumo, "-c", cfg]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true", help="run with SUMO GUI")
    ap.add_argument("--cfg", type=str, default="config.sumocfg", help="sumo config file")
    args = ap.parse_args()

    cmd = build_cmd(args.gui, args.cfg)
    print("Launching:", " ".join(cmd), file=sys.stderr)

    traci.start(cmd)
    step = 0
    try:
        # Step at least once; then continue while vehicles are expected or until 3600s (fallback)
        end_t = 3600.0
        while True:
            traci.simulationStep()
            step += 1
            t = traci.simulation.getTime()
            # vehicles still in the network or scheduled to depart
            if traci.simulation.getMinExpectedNumber() == 0 and t >= 1.0:
                break
            if t >= end_t:
                break
        print(f"Finished stepping at t={traci.simulation.getTime():.2f}s, steps={step}", file=sys.stderr)
    finally:
        traci.close()

if __name__ == "__main__":
    main()
