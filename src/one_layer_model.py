import numpy as np
import os


class OneLayerModel:

    def __init__(self, sigma, name="one-layer-model") -> None:
        self.sigma = sigma
        self._name = name

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    @property
    def name(self):
        return self._name

    def one_over_root(self, r, z):
        return 1 / (np.sqrt(np.power(r, 2) + np.power(z, 2)))

    def field_potential(self, I, r, z):
        q = I / (2 * np.pi * self.sigma)
        return q * (self.one_over_root(r, z))

    def field_potential_surface(self, I, r):
        return self.field_potential(I, r, 0)

    def one_over_root_dr(self, r, z):
        a = np.power(r, 2)
        b = np.power(z, 2)
        return r / (np.power(a + b, 3 / 2))

    def one_over_root_dz(self, r, z):
        a = np.power(r, 2)
        b = np.power(z, 2)
        return z / (np.power(a + b, 3 / 2))

    def field_strength(self, I, r, z):
        q = I / (2 * np.pi * self.sigma)
        mdVdr = q * self.one_over_root_dr(r, z)
        mdVdz = q * self.one_over_root_dz(r, z)
        return (mdVdr, mdVdz)


if __name__ == "__main__":
    print("One Layer Model")
    one_layer_model = OneLayerModel(1)
    r_coords = np.linspace(-1, 1, 51, dtype=np.float64)
    z_coords = np.linspace(0, 1, 50, dtype=np.float64)
    I = 1
    r, z = np.meshgrid(r_coords, z_coords)

    v = one_layer_model.field_potential(I, r, z)
    er, ez = one_layer_model.field_strength(I, r, z)

    os.makedirs("./computed/{}".format(one_layer_model.name), exist_ok=True)
    np.savetxt(
        "./computed/{}/potential.csv".format(one_layer_model.name), v, delimiter=","
    )
    np.savetxt(
        "./computed/{}/strength_r.csv".format(one_layer_model.name), er, delimiter=","
    )
    np.savetxt(
        "./computed/{}/strength_z.csv".format(one_layer_model.name), ez, delimiter=","
    )
    parameters = {"sigma": one_layer_model.sigma, "I": I}
    np.save("./computed/{}/parameters.npy".format(one_layer_model.name), parameters)

    np.savetxt("./computed/{}/r.csv".format(one_layer_model.name),
               r, delimiter=",")
    np.savetxt("./computed/{}/z.csv".format(one_layer_model.name),
               z, delimiter=",")
