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

    def inf_sum(self, r, z):
        return mpmath.nsum(lambda n: (self.k_12 ** n) *
                    (self.one_over_root(r, -1 * z, n) + self.one_over_root(r, z, n)), [1, mpmath.inf])

    def field_potential(self, I, r, z):
        q = I / (2*mpmath.pi * self.sigma_1)
        return q * (self.one_over_root(r, z, 0) + self.inf_sum(r, z))

    def field_potential_surface(self, I, r):
        return self.field_potential(I, r, 0)


if __name__ == "__main__":
    print("Two Layer Model")
    a = TwoLayerModel(1, 0.5, 1)
    r_coords = mpmath.linspace(-10, 10, 20)
    z_coords = mpmath.linspace(0, 10, 20)
    I = 1
    r, z = np.meshgrid(r_coords, z_coords)
    line = []
    v = []
    for rr in r_coords:
        for zz in z_coords:
            line.append(a.field_potential(I, rr, zz))
        v.append(line)
        line = []

    v = np.array(v, dtype=float)

    plt.contour(r, z, v, colors='black')
    plt.show()

