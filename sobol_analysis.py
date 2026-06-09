import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from SALib.sample import saltelli
from SALib.analyze import sobol
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════
#  LOAD TRAINED GPR MODELS (best accuracy)
# ══════════════════════════════════════════════
print("Loading trained models...")
with open('trained_models.pkl', 'rb') as f:
    saved = pickle.load(f)

models = saved['models']
scaler = saved['scaler']

gpr_def = models['GPR']['deflection']
gpr_str = models['GPR']['stress']

print("GPR models loaded.")

# ══════════════════════════════════════════════
#  DEFINE PARAMETER SPACE
# ══════════════════════════════════════════════
problem = {
    'num_vars': 3,
    'names': ['Vf', 'theta', 'F'],
    'bounds': [
        [0.3,  0.7],    # Fiber volume fraction
        [0.0,  90.0],   # Fiber orientation (degrees)
        [500,  5000],   # Applied load (N)
    ]
}

# ══════════════════════════════════════════════
#  GENERATE SALTELLI SAMPLES
# ══════════════════════════════════════════════
N = 1024  # base sample size — total = N*(2D+2) = 8192 samples
print(f"\nGenerating Saltelli samples (N={N})...")
param_values = saltelli.sample(problem, N, calc_second_order=True)
print(f"Total samples: {param_values.shape[0]}")

# ══════════════════════════════════════════════
#  EVALUATE SURROGATE ON SAMPLES
# ══════════════════════════════════════════════
print("Evaluating GPR surrogate...")
X_scaled = scaler.transform(param_values)

Y_def    = np.exp(gpr_def.predict(X_scaled))
Y_stress = np.exp(gpr_str.predict(X_scaled))

print(f"Deflection range : {Y_def.min():.4e} — {Y_def.max():.4e} m")
print(f"Stress range     : {Y_stress.min():.4e} — {Y_stress.max():.4e} Pa")

# ══════════════════════════════════════════════
#  SOBOL' ANALYSIS
# ══════════════════════════════════════════════
print("\nRunning Sobol' analysis...")

Si_def    = sobol.analyze(problem, Y_def,    calc_second_order=True)
Si_stress = sobol.analyze(problem, Y_stress, calc_second_order=True)

# ══════════════════════════════════════════════
#  PRINT RESULTS
# ══════════════════════════════════════════════
names = problem['names']

print("\n" + "=" * 60)
print("  SOBOL' SENSITIVITY — TIP DEFLECTION")
print("=" * 60)
print(f"{'Parameter':<10} {'S1':>8} {'S1_conf':>10} "
      f"{'ST':>8} {'ST_conf':>10}")
print("-" * 60)
for i, name in enumerate(names):
    print(f"{name:<10} {Si_def['S1'][i]:>8.4f} "
          f"{Si_def['S1_conf'][i]:>10.4f} "
          f"{Si_def['ST'][i]:>8.4f} "
          f"{Si_def['ST_conf'][i]:>10.4f}")

print("\n" + "=" * 60)
print("  SOBOL' SENSITIVITY — MAX VON MISES STRESS")
print("=" * 60)
print(f"{'Parameter':<10} {'S1':>8} {'S1_conf':>10} "
      f"{'ST':>8} {'ST_conf':>10}")
print("-" * 60)
for i, name in enumerate(names):
    print(f"{name:<10} {Si_stress['S1'][i]:>8.4f} "
          f"{Si_stress['S1_conf'][i]:>10.4f} "
          f"{Si_stress['ST'][i]:>8.4f} "
          f"{Si_stress['ST_conf'][i]:>10.4f}")

# Second order indices
print("\n" + "=" * 60)
print("  SECOND ORDER INDICES — TIP DEFLECTION")
print("=" * 60)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        print(f"  S2({names[i]}, {names[j]}) = "
              f"{Si_def['S2'][i][j]:.4f} "
              f"± {Si_def['S2_conf'][i][j]:.4f}")

print("\n" + "=" * 60)
print("  SECOND ORDER INDICES — MAX STRESS")
print("=" * 60)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        print(f"  S2({names[i]}, {names[j]}) = "
              f"{Si_stress['S2'][i][j]:.4f} "
              f"± {Si_stress['S2_conf'][i][j]:.4f}")

# ══════════════════════════════════════════════
#  SAVE RESULTS
# ══════════════════════════════════════════════
results = {
    'deflection': {
        'S1':      Si_def['S1'].tolist(),
        'ST':      Si_def['ST'].tolist(),
        'S1_conf': Si_def['S1_conf'].tolist(),
        'ST_conf': Si_def['ST_conf'].tolist(),
    },
    'stress': {
        'S1':      Si_stress['S1'].tolist(),
        'ST':      Si_stress['ST'].tolist(),
        'S1_conf': Si_stress['S1_conf'].tolist(),
        'ST_conf': Si_stress['ST_conf'].tolist(),
    },
    'names': names
}

pd.DataFrame({
    'parameter'  : names,
    'S1_def'     : Si_def['S1'],
    'ST_def'     : Si_def['ST'],
    'S1_stress'  : Si_stress['S1'],
    'ST_stress'  : Si_stress['ST'],
}).to_csv('sobol_results.csv', index=False)
print("\nResults saved: sobol_results.csv")

# ══════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

colors_s1 = ['#2196F3', '#4CAF50', '#FF9800']
colors_st = ['#1565C0', '#2E7D32', '#E65100']

# ── Plot 1: S1 — Deflection ──
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(names, Si_def['S1'], color=colors_s1,
               yerr=Si_def['S1_conf'], capsize=6,
               edgecolor='white', linewidth=0.8)
ax1.set_title('First-Order Indices (S1)\nTip Deflection',
              fontweight='bold')
ax1.set_ylabel('Sobol Index')
ax1.set_ylim(0, 1.05)
ax1.grid(True, axis='y', alpha=0.3)
ax1.axhline(y=0, color='black', linewidth=0.5)
for bar, val in zip(bars, Si_def['S1']):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

# ── Plot 2: ST — Deflection ──
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(names, Si_def['ST'], color=colors_st,
               yerr=Si_def['ST_conf'], capsize=6,
               edgecolor='white', linewidth=0.8)
ax2.set_title('Total-Order Indices (ST)\nTip Deflection',
              fontweight='bold')
ax2.set_ylabel('Sobol Index')
ax2.set_ylim(0, 1.05)
ax2.grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars, Si_def['ST']):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

# ── Plot 3: S1 vs ST — Deflection ──
ax3 = fig.add_subplot(gs[0, 2])
x   = np.arange(len(names))
w   = 0.35
ax3.bar(x - w/2, Si_def['S1'], w, label='S1 (direct)',
        color=colors_s1, edgecolor='white')
ax3.bar(x + w/2, Si_def['ST'], w, label='ST (total)',
        color=colors_st, edgecolor='white', alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(names)
ax3.set_title('S1 vs ST Comparison\nTip Deflection', fontweight='bold')
ax3.set_ylabel('Sobol Index')
ax3.set_ylim(0, 1.1)
ax3.legend()
ax3.grid(True, axis='y', alpha=0.3)

# ── Plot 4: S1 — Stress ──
ax4 = fig.add_subplot(gs[1, 0])
bars = ax4.bar(names, Si_stress['S1'], color=colors_s1,
               yerr=Si_stress['S1_conf'], capsize=6,
               edgecolor='white', linewidth=0.8)
ax4.set_title('First-Order Indices (S1)\nMax Von Mises Stress',
              fontweight='bold')
ax4.set_ylabel('Sobol Index')
ax4.set_ylim(0, 1.05)
ax4.grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars, Si_stress['S1']):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

# ── Plot 5: ST — Stress ──
ax5 = fig.add_subplot(gs[1, 1])
bars = ax5.bar(names, Si_stress['ST'], color=colors_st,
               yerr=Si_stress['ST_conf'], capsize=6,
               edgecolor='white', linewidth=0.8)
ax5.set_title('Total-Order Indices (ST)\nMax Von Mises Stress',
              fontweight='bold')
ax5.set_ylabel('Sobol Index')
ax5.set_ylim(0, 1.05)
ax5.grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars, Si_stress['ST']):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

# ── Plot 6: S1 vs ST — Stress ──
ax6 = fig.add_subplot(gs[1, 2])
ax6.bar(x - w/2, Si_stress['S1'], w, label='S1 (direct)',
        color=colors_s1, edgecolor='white')
ax6.bar(x + w/2, Si_stress['ST'], w, label='ST (total)',
        color=colors_st, edgecolor='white', alpha=0.8)
ax6.set_xticks(x)
ax6.set_xticklabels(names)
ax6.set_title('S1 vs ST Comparison\nMax Von Mises Stress',
              fontweight='bold')
ax6.set_ylabel('Sobol Index')
ax6.set_ylim(0, 1.1)
ax6.legend()
ax6.grid(True, axis='y', alpha=0.3)

fig.suptitle(
    "Sobol' Sensitivity Analysis — NACA 0015 Composite Blade\n"
    "GPR Surrogate | 8192 Samples",
    fontsize=14, fontweight='bold', y=1.01)

plt.savefig('sobol_results.png', dpi=600, bbox_inches='tight')
print("Plot saved: sobol_results.png")
print("\nAll done!")