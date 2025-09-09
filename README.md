# Guided vs. Free Parking — SUMO Template

This is a **ready-to-run SUMO template** for studying *guided (assigned) parking* vs. *free search* with
special spaces (Family / Disabled), compliance, enforcement, info delay, and dynamic reassignment.

> Tested with SUMO >= 1.18. Adjust lane IDs/positions if you modify the network geometry.

---

## Quick start

1. **Install SUMO** (https://www.eclipse.org/sumo/), ensure `sumo`, `sumo-gui`, and `netgenerate` are in PATH.
2. (Optional) Create a Python env and install `traci`:
   ```bash
   pip install eclipse-sumo
   ```
3. **Run (GUI)**:
   ```bash
   python controller.py --gui
   ```
   or **run (headless)**:
   ```bash
   python controller.py
   ```

The controller will launch SUMO with `config.sumocfg`, spawn traffic (normal/family/disabled),
assign vehicles to legal parking areas, handle dynamic re-assignment, basic enforcement and KPI logging.

---

## File structure

```
parking_guided_template/
  ├─ config.sumocfg          # SUMO configuration
  ├─ network.net.xml         # Prebuilt mini network (simple aisles loop)
  ├─ routes.rou.xml          # Vehicle types and demand (flows)
  ├─ parking.add.xml         # Parking areas: General / Family / Disabled
  ├─ controller.py           # TraCI controller (assignment, reassignment, enforcement)
  ├─ params.yaml             # Tunable parameters (shares, delays, compliance, etc.)
  └─ README.md               # This file
```

> If you change the network, update `parking.add.xml` lane references and `positions` accordingly.

---

## What’s implemented

- **User types**: `normal`, `family`, `disabled`
- **Special spaces**: `PA_F_*` (Family), `PA_D_*` (Disabled), plus general `PA_G_*`
- **Eligibility rules** (hard constraint): normal users **cannot** park in special areas
- **Guidance strategies**:
  - `free` (baseline): users choose by a utility function (distance + walk + congestion)
  - `assign_static`: assigned once on entry
  - `assign_dynamic`: assigned on entry and **one** possible reassignment if target becomes unavailable
- **Compliance**: configurable `p_compliance` (≤ 1.0); non-compliant cars fall back to free choice
- **Info delay / detection noise**: configurable polling interval and (optional) inaccuracy
- **Enforcement** (optional): sweep special areas; detect violations with probability; reassign/penalize
- **KPIs logged** to `output/kpis.csv`: search time, cruising distance, fuel/CO2 (if emission model enabled),
  queue length proxies, occupancy, fairness metrics (success rates for special users)

---

## Tuning

Edit `params.yaml`:
- `scenario`: `"free" | "assign_static" | "assign_dynamic"`
- shares of user types, special space ratio, compliance, info delay, detection rate, etc.
- demand level (`vehPerHour`) and occupancy target (by run length + capacity)

---

## Notes

- The **network** is a simple loop with two inner aisles to attach parking areas; adapt to your geometry.
- **Parking area positions** are given in meters along the lane; adjust or add more blocks as needed.
- For **emissions/energy**, enable SUMO’s HBEFA or PHEMlight models (`--emission-output` or in config).
- Use additional scripts to **aggregate by seeds** and run **factorial experiments**.
