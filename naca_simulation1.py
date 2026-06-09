from fenics import *
import numpy as np

set_log_level(30)  # suppress FEniCS output

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

    V  = VectorFunctionSpace(mesh, 'P', 1)

    # ── Boundary condition: fix leading edge (x near 0) ──
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

    # ── Variational problem ──
    u, v   = TrialFunction(V), TestFunction(V)
    f_body = Constant((0, 0))
    pressure = Constant((0, -F))

    a      = inner(sigma_composite(u), sym(nabla_grad(v)))*dx
    L_form = dot(f_body, v)*dx + dot(pressure, v)*ds

    # ── Solve ──
    u_sol = Function(V)
    solve(a == L_form, u_sol, bc)

    # ── Extract outputs ──
    tip_deflection = abs(u_sol.vector().min())

    W  = FunctionSpace(mesh, 'P', 1)
    VM = project(
        sqrt(  sigma_composite(u_sol)[0,0]**2
             - sigma_composite(u_sol)[0,0]*sigma_composite(u_sol)[1,1]
             + sigma_composite(u_sol)[1,1]**2
             + 3*sigma_composite(u_sol)[0,1]**2), W)
    max_stress = VM.vector().max()

    return tip_deflection, max_stress


if __name__ == "__main__":
    print("Running NACA 0015 airfoil simulation...")
    print("Parameters: Vf=0.5, theta=0°, F=1000 N")
    print("-" * 45)

    tip, stress = run_airfoil_simulation(Vf=0.5, theta=0.0, F=1000.0)

    print(f"Tip deflection : {tip:.6f} m")
    print(f"Max stress     : {stress/1e6:.2f} MPa")
    print("-" * 45)
    print("Airfoil simulation working!")