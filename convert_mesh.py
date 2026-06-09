import gmsh
import numpy as np
import meshio

def naca0015(t, c=1.0, n=50):
    x = np.linspace(0, c, n)
    yt = 5*t * (0.2969*np.sqrt(x/c)
               - 0.1260*(x/c)
               - 0.3516*(x/c)**2
               + 0.2843*(x/c)**3
               - 0.1015*(x/c)**4)
    x_upper = x
    y_upper = yt
    x_lower = x[::-1]
    y_lower = -yt[::-1]
    x_coords = np.concatenate([x_upper, x_lower[1:]])
    y_coords = np.concatenate([y_upper, y_lower[1:]])
    return x_coords, y_coords

# ── Generate mesh with gmsh ──
gmsh.initialize()
gmsh.model.add("naca0015")

x, y = naca0015(0.15)
point_tags = []
for xi, yi in zip(x, y):
    tag = gmsh.model.geo.addPoint(xi, yi, 0, meshSize=0.02)
    point_tags.append(tag)

line_tags = []
n = len(point_tags)
for i in range(n):
    tag = gmsh.model.geo.addLine(point_tags[i], point_tags[(i+1) % n])
    line_tags.append(tag)

loop    = gmsh.model.geo.addCurveLoop(line_tags)
surface = gmsh.model.geo.addPlaneSurface([loop])
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(2, [surface], tag=1)
gmsh.model.mesh.generate(2)
gmsh.write("airfoil.msh")
gmsh.finalize()
print("Gmsh mesh generated: airfoil.msh")

# ── Convert to XDMF using meshio ──
mesh = meshio.read("airfoil.msh")

# Extract only triangle cells
cells  = [c for c in mesh.cells if c.type == "triangle"]
points = mesh.points[:, :2]  # drop z coordinate

meshio.write(
    "airfoil.xdmf",
    meshio.Mesh(points=points, cells=cells)
)
print("Converted to XDMF: airfoil.xdmf")