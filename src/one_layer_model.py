import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class OneLayerModel:

    def __init__(self, sigma) -> None:
        self.sigma = sigma

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    def one_over_root(self, r, z):
        return 1/(np.sqrt(np.power(r, 2) + np.power(z, 2)))

    def field_potential(self, I, r, z):
        q = I / (2*np.pi * self.sigma)
        return q * (self.one_over_root(r, z))

    def field_potential_surface(self, I, r):
        return self.field_potential(I, r, 0)

    def one_over_root_dr(self, r, z):
        a = np.power(r, 2)
        b = np.power(z, 2)
        return r/(np.power(a + b, 3/2))

    def one_over_root_dz(self, r, z):
        a = np.power(r, 2)
        b = np.power(z, 2)
        return z/(np.power(a + b, 3/2))

    def field_strength(self, I, r, z):
        q = I / (2*np.pi * self.sigma)
        mdVdr = q * self.one_over_root_dr(r, z)
        mdVdz = q * self.one_over_root_dz(r, z)
        return (mdVdr, mdVdz)


if __name__ == "__main__":
    print("One Layer Model")
    a = OneLayerModel(1)
    r_coords = np.linspace(-1, 1, 51, dtype=np.float128)
    z_coords = np.linspace(0, 1, 50, dtype=np.float128)
    I = 1
    r, z = np.meshgrid(r_coords, z_coords)

    v = a.field_potential(I, r, z)
    v = np.clip(v, 0, 1)

    er, ez = a.field_strength(I, r, z)
    norm = np.sqrt(np.add(np.power(er, 2), np.power(ez, 2)))
    er = er / norm
    ez = ez / norm

    figure = plt.figure()

    ax = figure.add_subplot()
    ax.invert_yaxis()

    cset_v = ax.contourf(r, z, v)
    cbi_v = figure.colorbar(cset_v, orientation='horizontal', shrink=0.8)
    cbi_v.set_label('Potential, V')

    n = 8
    M = np.hypot(er[1::n, 1::n], ez[1::n, 1::n])
    q_e = ax.quiver(r[1::n, 1::n], z[1::n, 1::n],
                    er[1::n, 1::n], -ez[1::n, 1::n], norm[1::n, 1::n], pivot="mid", cmap="gist_gray",
                    units='width')
    q_e.set_clim(0, 2)
    cbi_e = figure.colorbar(q_e, orientation='horizontal', shrink=0.8)
    cbi_e.set_label('Strength, V/m')

    plt.show()
