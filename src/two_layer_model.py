import mpmath
import numpy as np
import os


class TwoLayerModel:

    def __init__(self, sigma_1, sigma_2, d_1, name="two-layer-model") -> None:
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.d_1 = d_1
        self._name = name

    @property
    def name(self):
        return self._name

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
        if r == 0:
            return 0
        a = np.power(r, 2)
        sz = np.add(z, 2 * n * self.d_1)
        b = np.power(sz, 2)
        return 1 / (np.sqrt(a + b))

    def inf_sum_potential(self, r, z):
        return mpmath.nsum(
            lambda n: (self.k_12**n)
            * (self.one_over_root(r, -1 * z, n) + self.one_over_root(r, z, n)),
            [1, mpmath.inf],
        )

    def field_potential(self, I, r, z):
        q = I / (2 * np.pi * self.sigma_1)
        return q * (self.one_over_root(r, z, 0) + self.inf_sum_potential(r, z))

    def field_potential_surface(self, I, r):
        return self.field_potential(I, r, 0)

    def over_root_dr(self, r, z, n):
        if r == 0:
            return 0
        a = np.power(r, 2)
        sz = np.add(z, 2 * n * self.d_1)
        b = np.power(sz, 2)
        return r * (np.power(a + b, -3 / 2))

    def inf_sum_strength_dr(self, r, z):
        return mpmath.nsum(
            lambda n: (self.k_12**n)
            * (self.over_root_dr(r, -1 * z, n) + self.over_root_dr(r, z, n)),
            [1, mpmath.inf],
        )

    def over_root_dz(self, r, z, n):
        if r == 0:
            return 0
        sz = np.add(z, 2 * n * self.d_1)
        return sz / (np.power(np.power(r, 2) + np.power(sz, 2), 3 / 2))

    def inf_sum_strength_dz(self, r, z):
        return mpmath.nsum(
            lambda n: (self.k_12**n)
            * (-1 * self.over_root_dz(r, -1 * z, n) + self.over_root_dz(r, z, n)),
            [1, mpmath.inf],
        )

    def field_strength(self, I, r, z):
        q = I / (2 * np.pi * self.sigma_1)
        mdVdr = q * (self.over_root_dr(r, z, 0) +
                     self.inf_sum_strength_dr(r, z))
        mdVdz = q * (self.over_root_dz(r, z, 0) +
                     self.inf_sum_strength_dz(r, z))
        return (mdVdr, mdVdz)


if __name__ == "__main__":
    print("Two Layer Model")
    import parameters
    sigma_1 = parameters.sigma_fat
    sigma_2 = parameters.sigma_muscle
    d_1 = parameters.d_1
    r_size = 10*d_1
    z_size = 10*d_1
    two_layer_model = TwoLayerModel(sigma_1, sigma_2, d_1)
    r_coords = np.linspace(-r_size, r_size, 301, dtype=np.float64)
    z_coords = np.linspace(0, z_size, 300, dtype=np.float64)
    I = 1 * 10**(-3)
    r, z = np.meshgrid(r_coords, z_coords)

    line = []
    v = []
    line_er = []
    er = []
    line_ez = []
    ez = []
    for zz in z_coords:
        for rr in r_coords:
            line.append(two_layer_model.field_potential(I, rr, zz))
            value = two_layer_model.field_strength(I, rr, zz)
            line_er.append(value[0])
            line_ez.append(value[1])
        v.append(line)
        er.append(line_er)
        ez.append(line_ez)
        line = []
        line_er = []
        line_ez = []

    v = np.array(v, np.float64)

    er = np.array(er, np.float64)
    ez = np.array(ez, np.float64)

    os.makedirs("./computed/{}".format(two_layer_model.name), exist_ok=True)
    np.savetxt(
        "./computed/{}/potential.csv".format(two_layer_model.name), v, delimiter=","
    )
    np.savetxt(
        "./computed/{}/strength_r.csv".format(two_layer_model.name), er, delimiter=","
    )
    np.savetxt(
        "./computed/{}/strength_z.csv".format(two_layer_model.name), ez, delimiter=","
    )
    parameters = np.asarray(
        [two_layer_model.sigma_1, two_layer_model.sigma_2, two_layer_model.d_1, I]
    )

    parameters = {"sigma_1": two_layer_model.sigma_1,
                  "sigma_2": two_layer_model.sigma_1, "d_1": two_layer_model.d_1, "I": I}
    np.save("./computed/{}/parameters.npy".format(two_layer_model.name), parameters)

    np.savetxt("./computed/{}/r.csv".format(two_layer_model.name),
               r, delimiter=",")
    np.savetxt("./computed/{}/z.csv".format(two_layer_model.name),
               z, delimiter=",")
