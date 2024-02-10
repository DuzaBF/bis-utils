import mpmath
import numpy as np
import matplotlib.pyplot as plt


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
        q = np.add(z, 2*n*self.d_1)
        b = np.power(q, 2)
        return 1/(mpmath.sqrt(a + b))

    def inf_sum_potential(self, r, z):
        return mpmath.nsum(lambda n: (self.k_12 ** n) *
                           (self.one_over_root(r, -1 * z, n) + self.one_over_root(r, z, n)), [1, mpmath.inf])

    def field_potential(self, I, r, z):
        q = I / (2*mpmath.pi * self.sigma_1)
        return q * (self.one_over_root(r, z, 0) + self.inf_sum_potential(r, z))

    def field_potential_surface(self, I, r):
        return self.field_potential(I, r, 0)

    def one_over_root_dr(self, r, z, n):
        a = np.power(r, 2)
        q = np.add(z, 2*n*self.d_1)
        b = np.power(q, 2)
        return r/(mpmath.power(a + b, 3/2))

    def inf_sum_strength_dr(self, r, z):
        return mpmath.nsum(lambda n: (self.k_12 ** n) *
                           (self.one_over_root_dr(r, -1 * z, n) + self.one_over_root_dr(r, z, n)), [1, mpmath.inf])

    def one_over_root_dz(self, r, z, n):
        return z / (np.power(np.power(r, 2) + 2*n*self.d_1 + np.power(z, 2), 3/2))

    def inf_sum_strength_dz(self, r, z):
        return mpmath.nsum(lambda n: (self.k_12 ** n) *
                           (self.one_over_root_dz(r, -1 * z, n) + self.one_over_root_dz(r, z, n)), [1, mpmath.inf])

    def field_strength(self, I, r, z):
        q = I / (2*mpmath.pi * self.sigma_1)
        mdVdr = q * (self.one_over_root_dr(r, z, 0) + self.inf_sum_strength_dr(r, z))
        mdVdz = q * (-self.inf_sum_strength_dr(r, z))
        return (mdVdr, mdVdz)

if __name__ == "__main__":
    print("Two Layer Model")
    a = TwoLayerModel(0.5, 2, 5)
    r_coords = np.linspace(1/100, 5, 50)
    z_coords = np.linspace(1, 10, 50)
    I = 1
    r, z = np.meshgrid(r_coords, z_coords)
    line_v = []
    line_er = []
    line_ez = []
    v = []
    er = []
    ez = []
    for rr in r_coords:
        for zz in z_coords:
            line_v.append(a.field_potential(I, rr, zz))
            value = a.field_strength(I, rr, zz)
            line_er.append(value[0])
            line_ez.append(value[1])
        v.append(line_v)
        er.append(line_er)
        ez.append(line_ez)
        line_v = []
        line_er = []
        line_ez = []

    v = np.array(v, dtype=float)
    er = np.array(er, dtype=float)
    ez = np.array(ez, dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(r, z, v)
    ax.set_xlabel("r")
    ax.set_ylabel("z")
    ax.set_zlabel("v")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.contour(r[:-1], z[:-1], v[:-1])
    ax2.set_xlabel("r")
    ax2.set_ylabel("z")
    ax2.plot(r_coords[:-1], a.d_1 * np.ones(r_coords[:-1].shape), color="black")
    ax2.plot(-r_coords[:-1], a.d_1 * np.ones(r_coords[:-1].shape), color="black")
    ax2.invert_yaxis()
    ax2.contour(-r[:-1], z[:-1], v[:-1])

    fige = plt.figure()
    axe = fige.add_subplot()
    axe.quiver(r[:-1], z[:-1], er[:-1], ez[:-1])

    plt.show()
