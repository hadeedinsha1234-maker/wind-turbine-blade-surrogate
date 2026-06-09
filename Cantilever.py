from fenics import *
import numpy as np

# ── Composite material parameters (you will loop over these later) ──
Vf    = 0.5       # Fiber volume fraction
theta = 45.0       # Fiber orientation angle (degrees)
F     = 1000.0    # Applied tip load (N)

# ── Micromechanics: Rule of Mixtures (E-glass/epoxy) ──
E_fiber  = 72e9;   nu_fiber  = 0.22
E_matrix = 3.5e9;  nu_matrix = 0.35

E1  = Vf*E_fiber  + (1-Vf)*E_matrix          # Longitudinal modulus
E2  = E_matrix / (1 - Vf*(1 - E_matrix/E_fiber))  # Transverse modulus
nu12 = Vf*nu_fiber + (1-Vf)*nu_matrix         # Major Poisson ratio
G12 = E_matrix / (2*(1 - Vf*(1 - E_matrix/(2*(1+nu_fiber)*E_matrix/(2*(1+nu_matrix))*E_fiber))))

print(f"E1={E1/1e9:.2f} GPa, E2={E2/1e9:.2f} GPa, nu12={nu12:.3f}")

# ── Rotate stiffness to global frame ──
t = np.radians(theta)
c, s = np.cos(t), np.sin(t)

Q11 = E1/(1 - nu12**2 * E2/E1)
Q22 = E2/(1 - nu12**2 * E2/E1)
Q12 = nu12*E2/(1 - nu12**2 * E2/E1)
Q66 = G12

# Transformed stiffness components
Q11b = Q11*c**4 + 2*(Q12+2*Q66)*s**2*c**2 + Q22*s**4
Q22b = Q11*s**4 + 2*(Q12+2*Q66)*s**2*c**2 + Q22*c**4
Q12b = (Q11+Q22-4*Q66)*s**2*c**2 + Q12*(s**4+c**4)
Q66b = (Q11+Q22-2*Q12-2*Q66)*s**2*c**2 + Q66*(s**4+c**4)
Q16b = (Q11-Q12-2*Q66)*c**3*s - (Q22-Q12-2*Q66)*s**3*c
Q26b = (Q11-Q12-2*Q66)*c*s**3 - (Q22-Q12-2*Q66)*s*c**3

# ── Mesh ──
mesh = RectangleMesh(Point(0,0), Point(1.0, 0.1), 30, 6)
V = VectorFunctionSpace(mesh, 'P', 1)

# ── Boundary condition: fixed at root ──
bc = DirichletBC(V, Constant((0,0)),
                 lambda x, on_b: on_b and near(x[0], 0.0))

# ── Composite stress using transformed Q matrix ──
def sigma_composite(u):
    eps = sym(nabla_grad(u))
    e11, e22 = eps[0,0], eps[1,1]
    e12 = eps[0,1]
    s11 = Q11b*e11 + Q12b*e22 + Q16b*2*e12
    s22 = Q12b*e11 + Q22b*e22 + Q26b*2*e12
    s12 = Q16b*e11 + Q26b*e22 + Q66b*2*e12
    return as_tensor([[s11, s12],[s12, s22]])

# ── Solve ──
u, v = TrialFunction(V), TestFunction(V)
a = inner(sigma_composite(u), sym(nabla_grad(v)))*dx
L = dot(Constant((0,0)), v)*dx + dot(Constant((0,-F)), v)*ds

u_sol = Function(V)
solve(a == L, u_sol, bc)

# ── Extract outputs ──
tip_deflection = abs(u_sol.vector().min())
max_disp       = u_sol.vector().max()

print(f"\nParameters: Vf={Vf}, theta={theta}°, F={F} N")
print(f"Tip deflection : {tip_deflection:.6f} m")
print(f"Max displacement: {max_disp:.6f} m")
print("Done!")