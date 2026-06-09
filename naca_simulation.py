from fenics import *
import numpy as np

set_log_level(30)

def run_airfoil_simulation(Vf, theta, F, mesh_file="airfoil.xdmf"):

    # ── Material properties (E-glass/epoxy) ──
    E_fiber  = 72e9;  nu_fiber  = 0.22
    E_matrix = 3.5e9; nu_matrix = 0.35

    E1   = Vf*E_fiber + (1-Vf)*E_matrix
    E2   = E_matrix / (1 - Vf*(1 - E_matrix/E_fiber))
    nu12 = Vf*nu_fiber + (1-Vf)*nu_matrix
    G12  = E_matrix / (2*(1+nu_matrix)) / (1 - Vf*(1 - E_matrix/(2*(1+nu_matrix)) / (E_fiber/(2*(1+nu_fiber)))))

    # ── Rotate stiffness matrix ──
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

    # ── Load mesh ──
    mesh = Mesh()
    with XDMFFile(mesh_file) as f:
        f.read(mesh)

    V = VectorFunctionSpace(mesh, 'P', 1)

    # ── Boundary condition: fix leading edge ──
    def leading_edge(x, on_boundary):
        return on_boundary and near(x[0], 0.0, 0.05)

    bc = DirichletBC(V, Constant((0, 0)), leading_edge)

    # ── Composite stress tensor ──
    def sigma_composite(u):
        eps = sym(nabla_grad(u))
        e11, e22, e12 = eps[0,0], eps[1,1], eps[0,1]
        s11 = Q11b*e11 + Q12b*e22 + Q16b*2*e12
        s22 = Q12b*e11 + Q22b*e22 + Q26b*2*e12
        s12 = Q16b*e11 + Q26b*e22 + Q66b*2*e12
        return as_tensor([[s11, s12],[s12, s22]])

    # ── Solve ──
    u, v   = TrialFunction(V), TestFunction(V)
    a      = inner(sigma_composite(u), sym(nabla_grad(v)))*dx
    L_form = dot(Constant((0,0)), v)*dx + dot(Constant((0,-F)), v)*ds

    u_sol = Function(V)
    solve(a == L_form, u_sol, bc)

    # ── Tip deflection ──
    tip_deflection = abs(u_sol.vector().min())

    # ── Von Mises stress (robust extraction) ──
    T       = TensorFunctionSpace(mesh, 'DG', 0)
    sig     = project(sigma_composite(u_sol), T)
    sig_arr = sig.vector().get_local().reshape(-1, 4)

    s11 = sig_arr[:, 0]
    s22 = sig_arr[:, 3]
    s12 = sig_arr[:, 1]

    vm = np.sqrt(np.abs(s11**2 - s11*s22 + s22**2 + 3*s12**2))
    vm_clean = vm[np.isfinite(vm)]

    if len(vm_clean) == 0:
        # Fallback: estimate from deflection
        max_stress = E1 * tip_deflection / 1.0
    else:
        max_stress = float(np.max(vm_clean))

    return tip_deflection, max_stress