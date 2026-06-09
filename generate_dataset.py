from fenics import *
import numpy as np
import pandas as pd
import itertools

set_log_level(30)  # suppress FEniCS output

def run_simulation(Vf, theta, F, L=1.0, h=0.1):
    # Material properties
    E_fiber  = 72e9;  nu_fiber  = 0.22
    E_matrix = 3.5e9; nu_matrix = 0.35

    E1   = Vf*E_fiber + (1-Vf)*E_matrix
    E2   = E_matrix / (1 - Vf*(1 - E_matrix/E_fiber))
    nu12 = Vf*nu_fiber + (1-Vf)*nu_matrix
    G12  = E_matrix / (2*(1+nu_matrix)) / (1 - Vf*(1 - E_matrix/(2*(1+nu_matrix)) / (E_fiber/(2*(1+nu_fiber)))))

    t = np.radians(theta)
    c, s = np.cos(t), np.sin(t)

    Q11 = E1/(1 - nu12**2*E2/E1)
    Q22 = E2/(1 - nu12**2*E2/E1)
    Q12 = nu12*E2/(1 - nu12**2*E2/E1)
    Q66 = G12

    Q11b = Q11*c**4 + 2*(Q12+2*Q66)*s**2*c**2 + Q22*s**4
    Q22b = Q11*s**4 + 2*(Q12+2*Q66)*s**2*c**2 + Q22*c**4
    Q12b = (Q11+Q22-4*Q66)*s**2*c**2 + Q12*(s**4+c**4)
    Q66b = (Q11+Q22-2*Q12-2*Q66)*s**2*c**2 + Q66*(s**4+c**4)
    Q16b = (Q11-Q12-2*Q66)*c**3*s - (Q22-Q12-2*Q66)*s**3*c
    Q26b = (Q11-Q12-2*Q66)*c*s**3 - (Q22-Q12-2*Q66)*s*c**3

    mesh = RectangleMesh(Point(0,0), Point(L,h), 30, 6)
    V    = VectorFunctionSpace(mesh, 'P', 1)
    bc   = DirichletBC(V, Constant((0,0)),
                       lambda x, on_b: on_b and near(x[0], 0.0))

    def sigma_c(u):
        eps = sym(nabla_grad(u))
        e11, e22, e12 = eps[0,0], eps[1,1], eps[0,1]
        s11 = Q11b*e11 + Q12b*e22 + Q16b*2*e12
        s22 = Q12b*e11 + Q22b*e22 + Q26b*2*e12
        s12 = Q16b*e11 + Q26b*e22 + Q66b*2*e12
        return as_tensor([[s11,s12],[s12,s22]])

    u, v   = TrialFunction(V), TestFunction(V)
    a      = inner(sigma_c(u), sym(nabla_grad(v)))*dx
    L_form = dot(Constant((0,0)), v)*dx + dot(Constant((0,-F)), v)*ds

    u_sol = Function(V)
    solve(a == L_form, u_sol, bc)

    tip_deflection = abs(u_sol.vector().min())

    # Von Mises stress
    W     = FunctionSpace(mesh, 'P', 1)
    stress = sigma_c(u_sol)
    VM    = project(sqrt(stress[0,0]**2 - stress[0,0]*stress[1,1] +
                         stress[1,1]**2 + 3*stress[0,1]**2), W)
    max_stress = VM.vector().max()

    return tip_deflection, max_stress

# ── Parameter ranges ──
Vf_vals    = np.linspace(0.3, 0.7, 5)      # 5 values
theta_vals = np.linspace(0, 90, 7)          # 7 values
F_vals     = np.linspace(500, 5000, 6)      # 6 values

total = len(Vf_vals) * len(theta_vals) * len(F_vals)
print(f"Running {total} simulations...\n")

results = []
count   = 0

for Vf, theta, F in itertools.product(Vf_vals, theta_vals, F_vals):
    tip_def, max_stress = run_simulation(Vf, theta, F)
    results.append({
        'Vf': round(Vf, 3),
        'theta': round(theta, 1),
        'F': round(F, 1),
        'tip_deflection': tip_def,
        'max_stress': max_stress
    })
    count += 1
    print(f"[{count}/{total}] Vf={Vf:.2f}, θ={theta:.0f}°, F={F:.0f}N → "
          f"δ={tip_def:.6f}m, σ={max_stress/1e6:.2f}MPa")

# ── Save ──
df = pd.DataFrame(results)
df.to_csv('blade_dataset.csv', index=False)
print(f"\nDataset saved: {len(df)} rows → blade_dataset.csv")