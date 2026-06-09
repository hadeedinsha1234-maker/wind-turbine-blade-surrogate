import gmsh
import numpy as np

def naca0015(t, c=1.0, n=50):
    """Generate NACA 0015 airfoil coordinates"""
    x = np.linspace(0, c, n)
    # NACA 4-digit thickness formula
    yt = 5*t * (0.2969*np.sqrt(x/c)
               - 0.1260*(x/c)
               - 0.3516*(x/c)**2
               + 0.2843*(x/c)**3
               - 0.1015*(x/c)**4)
    # Upper and lower surfaces
    x_upper = x
    y_upper = yt
    x_lower = x[::-1]
    y_lower = -yt[::-1]
    # Combine (closed loop)
    x_coords = np.concatenate([x_upper, x_lower[1:]])
    y_coords = np.concatenate([y_upper, y_lower[1:]])
    return x_coords, y_coords

def create_airfoil_mesh(filename="airfoil.msh", chord=1.0, thickness=0.15):
    gmsh.initialize()
    gmsh.model.add("naca0015")

    x, y = naca0015(thickness, chord)

    # Add points
    point_tags = []
    for xi, yi in zip(x, y):
        tag = gmsh.model.geo.addPoint(xi, yi, 0, meshSize=0.02)
        point_tags.append(tag)

    # Add lines connecting points
    line_tags = []
    n = len(point_tags)
    for i in range(n):
        tag = gmsh.model.geo.addLine(point_tags[i], point_tags[(i+1) % n])
        line_tags.append(tag)

    # Create surface
    loop = gmsh.model.geo.addCurveLoop(line_tags)
    surface = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    # Add physical groups for boundary conditions
    gmsh.model.addPhysicalGroup(1, line_tags[:25], tag=1)   # leading edge region
    gmsh.model.addPhysicalGroup(1, line_tags[25:], tag=2)   # trailing edge region
    gmsh.model.addPhysicalGroup(2, [surface], tag=1)         # surface

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()
    print(f"Mesh saved: {filename}")

if __name__ == "__main__":
    create_airfoil_mesh()