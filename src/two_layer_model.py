import mpmath
import numpy as np
import matplotlib.pyplot as plt

mpmath.dps = 15
mpmath.pretty = True


class TwoLayerModel:

    def __init__(self, sigma_1, sigma_2, d_1) -> None:
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.d_1 = d_1

    @property
    def sigma_1(self):
        return self._sigma_1

    @sigma_1.setter
    def sigma_1(self, sigma):
        self._sigma_1 = sigma

    @property
    def sigma_2(self):
        return self._sigma_2

    @sigma_2.setter
    def sigma_2(self, sigma):
        self._sigma_2 = sigma

    @property
    def k_12(self):
        return (self.sigma_1 - self.sigma_2) / (self.sigma_1 + self.sigma_2)

    @property
    def d_1(self):
        return self._d_1

    @d_1.setter
    def d_1(self, d):
        self._d_1 = d

    def one_over_root(self, r, z, n):
        a = np.power(r, 2)
        sz = np.add(z, 2*n*self.d_1)
        b = np.power(sz, 2)
        return 1/(np.sqrt(a + b))

    def inf_sum_potential(self, r, z):
        return mpmath.nsum(lambda n: (self.k_12 ** n) *
                           (self.one_over_root(r, -1 * z, n) + self.one_over_root(r, z, n)), [1, mpmath.inf])

    def field_potential(self, I, r, z):
        q = I / (2*np.pi * self.sigma_1)
        return q * (self.one_over_root(r, z, 0) + self.inf_sum_potential(r, z))

    def field_potential_surface(self, I, r):
        return self.field_potential(I, r, 0)

    def over_root_dr(self, r, z, n):
        a = np.power(r, 2)
        sz = np.add(z, 2*n*self.d_1)
        b = np.power(sz, 2)
        return r * (np.power(a + b, -3/2))

    def inf_sum_strength_dr(self, r, z):
        return mpmath.nsum(lambda n: (self.k_12 ** n) *
                           (self.over_root_dr(r, -1 * z, n) + self.over_root_dr(r, z, n)), [1, mpmath.inf])

    def over_root_dz(self, r, z, n):
        sz = np.add(z, 2*n*self.d_1)
        return sz / (np.power(np.power(r, 2) + np.power(sz, 2), 3/2))

    def inf_sum_strength_dz(self, r, z):
        return mpmath.nsum(lambda n: (self.k_12 ** n) *
                           (-1 * self.over_root_dz(r, -1 * z, n) + self.over_root_dz(r, z, n)), [1, mpmath.inf])

    def field_strength(self, I, r, z):
        q = I / (2*np.pi * self.sigma_1)
        mdVdr = q * (self.over_root_dr(r, z, 0) +
                     self.inf_sum_strength_dr(r, z))
        mdVdz = q * (self.over_root_dz(r, z, 0) +
                     self.inf_sum_strength_dz(r, z))
        return (mdVdr, mdVdz)


if __name__ == "__main__":
    print("Two Layer Model")
    a = TwoLayerModel(1, 4, 0.2)
    r_coords = np.linspace(-1, 1, 51, dtype=np.float64)
    z_coords = np.linspace(0, 1, 50, dtype=np.float64)
    I = 1
    r, z = np.meshgrid(r_coords, z_coords)

    line = []
    v = []
    line_er = []
    er = []
    line_ez = []
    ez = []
    for zz in z_coords:
        for rr in r_coords:
            line.append(a.field_potential(I, rr, zz))
            # line.append(rr+zz)
            value = a.field_strength(I, rr, zz)
            line_er.append(value[0])
            line_ez.append(value[1])
        v.append(line)
        er.append(line_er)
        ez.append(line_ez)
        line = []
        line_er = []
        line_ez = []

    v = np.array(v, np.float64)
    v = np.clip(v, 0, 0.5)

    er = np.array(er, np.float64)
    ez = np.array(ez, np.float64)
    norm = np.sqrt(np.add(np.power(er, 2), np.power(ez, 2)))
    er = er / norm
    ez = ez / norm

    figure = plt.figure()

    ax = figure.add_subplot()
    ax.invert_yaxis()
    ax.set_title(r"Electric field for one-layer model for I={} A, $\sigma_1$={} Sm/m, $\sigma_2$={} Sm/m, $d_1$={} m".format(I, a.sigma_1, a.sigma_2, a.d_1))
    ax.set_xlabel("r, m")
    ax.set_ylabel("z, m")

    cset_v = ax.contourf(r, z, v)
    cbi_v = figure.colorbar(cset_v, orientation='horizontal', shrink=0.8)
    cbi_v.set_label('Potential, V')
    brd = ax.plot(r_coords, a.d_1 * np.ones(r_coords.shape), color="black")

    n = 4
    q_e = ax.quiver(r[1::n, 1::n], z[1::n, 1::n],
                    er[1::n, 1::n], -ez[1::n, 1::n], norm[1::n, 1::n], pivot="mid", cmap="gist_gray",
                    units="xy",
                    width=0.008,
                    headwidth=2,
                    headlength=3,
                    headaxislength=3,
                    scale_units="xy",
                    scale=10
                    )
    q_e.set_clim(0, 2)
    cbi_e = figure.colorbar(q_e, orientation='horizontal', shrink=0.8)
    cbi_e.set_label('Strength, V/m')

    plt.show()
