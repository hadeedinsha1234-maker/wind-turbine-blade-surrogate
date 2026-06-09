from fenics import *
import numpy as np
import pandas as pd
import itertools
from naca_simulation import run_airfoil_simulation

set_log_level(30)

# ── Parameter ranges (1000 simulations) ──
Vf_vals    = np.linspace(0.3, 0.7, 10)
theta_vals = np.linspace(0, 90, 10)
F_vals     = np.linspace(500, 5000, 10)

total = len(Vf_vals) * len(theta_vals) * len(F_vals)
print(f"Total simulations: {total}")
print(f"Estimated time   : {total*7//60} hrs {total*7%60} mins")
print("-" * 60)

results = []
count   = 0
failed  = 0

for Vf, theta, F in itertools.product(Vf_vals, theta_vals, F_vals):
    try:
        tip_def, max_stress = run_airfoil_simulation(Vf, theta, F)

        # Validate outputs
        if not np.isfinite(tip_def) or not np.isfinite(max_stress):
            raise ValueError(f"Non-finite output: δ={tip_def}, σ={max_stress}")

        results.append({
            'Vf'            : round(float(Vf), 4),
            'theta'         : round(float(theta), 2),
            'F'             : round(float(F), 2),
            'tip_deflection': float(tip_def),
            'max_stress'    : float(max_stress)
        })
        count += 1
        print(f"[{count}/{total}] Vf={Vf:.2f}, θ={theta:.1f}°, "
              f"F={F:.0f}N → δ={tip_def:.6f}m, σ={max_stress/1e6:.3f}MPa")

        # Save checkpoint every 50 runs
        if count % 50 == 0:
            pd.DataFrame(results).to_csv('airfoil_dataset.csv', index=False)
            print(f"\n  ✓ Checkpoint saved: {count}/{total} rows\n")

    except Exception as e:
        failed += 1
        print(f"  [FAILED #{failed}] Vf={Vf:.2f}, "
              f"θ={theta:.1f}°, F={F:.0f}N → {e}")

# ── Final save ──
df = pd.DataFrame(results)
df.to_csv('airfoil_dataset.csv', index=False)
print("\n" + "=" * 60)
print(f"  DONE: {count} successful, {failed} failed")
print(f"  Dataset saved → airfoil_dataset.csv")
print("=" * 60)