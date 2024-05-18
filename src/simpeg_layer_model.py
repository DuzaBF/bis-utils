import numpy as np
import matplotlib.pyplot as plt
import SimPEG
from SimPEG import SolverLU as Solver

R_MESH_SIZE = 200
Z_MESH_SIZE = 100

# Mesh parameters
npad = 20
cs = 0.5
hx = [(cs, npad, -1.3), (cs, R_MESH_SIZE), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, Z_MESH_SIZE)]
mesh = SimPEG.discretize.TensorMesh([hx, hy], "CN")
print(mesh)

def get_potential(sigma_1, sigma_2, d_1):
    conductivity_map = sigma_2 * np.ones(mesh.n_cells)
    conductivity_map[mesh.cell_centers[:,1] >= -d_1] = sigma_1 # Set the conductivity of the first layer

    source1 = np.r_[0.0, 0.0, 0.0]
    source2 = np.r_[0.0, 0.0, 0.0]

    q = np.zeros(mesh.n_cells)
    a = mesh.closest_points_index(source1[:2])
    b = mesh.closest_points_index(source2[:2])

    q[a] = 1.0 / mesh.cell_volumes[a]
    q[a] = -1.0 / mesh.cell_volumes[b]

    A = (
        mesh.cell_gradient.T
        * SimPEG.discretize.utils.sdiag(1.0 / (mesh.dim * mesh.average_face_to_cell.T * (1.0 / conductivity_map)))
        * mesh.cell_gradient
    )
    print(A)

    Ainv = Solver(A)

    print(Ainv)

    V = Ainv * q

    print(V)

    return V

def plot_potential(sigma_1, sigma_2, d_1):
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    r_size = 10*d_1
    z_size = 10*d_1

    r = np.linspace(-r_size, r_size, R_MESH_SIZE)
    z = np.linspace(-z_size, 0, Z_MESH_SIZE)

    pltgrid = SimPEG.discretize.utils.ndgrid(r, z)
    rplt = pltgrid[:, 0].reshape(r.size, z.size, order="F")
    zplt = pltgrid[:, 1].reshape(r.size, z.size, order="F")

    Pc = mesh.get_interpolation_matrix(pltgrid, "CC")
    potential = get_potential(sigma_1, sigma_2, d_1)
    Vplt = Pc * potential
    Vplt = Vplt.reshape(r.size, z.size, order="F")

    ax.contour(rplt, zplt, Vplt)
    mesh.plot_grid(ax=ax2)
    plt.show()

if __name__ == "__main__":
    print("Two Layer Model")
    import parameters

    sigma_1 = parameters.sigma_fat
    sigma_2 = parameters.sigma_muscle
    d_1 = parameters.d_1

    plot_potential(sigma_1, sigma_2, d_1)