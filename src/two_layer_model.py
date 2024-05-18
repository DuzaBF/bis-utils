import mpmath
import numpy as np
import os
import time 

EPS = 1e-3


class TwoLayerModel:

    def __init__(self, sigma_1, sigma_2, d_1, name="two-layer-model") -> None:
        self.sigma_1 = np.float64(sigma_1)
        self.sigma_2 = np.float64(sigma_2)
        self.d_1 = np.float64(d_1)
        self._name = name
        self.debug = False

    def log(self, log_str):
        if self.debug:
            print(log_str)

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
        a = np.power(r, 2)
        sz = np.add(z, 2 * n * self.d_1)
        b = np.power(sz, 2)
        if np.isclose(np.float64(a), 0) and np.isclose(np.float64(b), 0):
            result = np.nan
        else:
            result = np.float64(np.power(a + b, -1 / 2))
        self.log(
            "one_over_root({:5.2f}, {:>5.2f}, {:d}) = {:3.2f}".format(
                r, z, int(n), result
            )
        )
        return result

    def sum_element(self, r, z, n):
        first_root = self.one_over_root(r, -1 * z, n)
        second_root = self.one_over_root(r, z, n)
        element = (np.power(self.k_12, n)) * (first_root + second_root)
        if self.debug:
            self.log(
                "element({:5.2f}, {:>5.2f}, {:d}) = {}".format(r, z, int(n), element)
            )
            self.running_total += element
            self.log("running total = {:>5.2f}".format(float(self.running_total)))
            self.a.append(first_root)
            self.b.append(second_root)
            self.c.append(element)
            self.d.append(self.running_total)
            self.e.append(np.power(self.k_12, n))
        return element

    def via_bessel(self, r, z, n, m):
        first = mpmath.exp(-2 * m * n * self.d_1)
        second = mpmath.exp(m * z) + mpmath.exp(-m * z)
        third = mpmath.besselj(0, m * r)
        return first * second * third
    
    def bessel_integral(self, r, z, n):
        f = lambda m: self.via_bessel(r, z, n, m)
        integral = mpmath.quad(f, [0, mpmath.inf], maxdegree=3)
        return integral

    def element_via_bessel(self, r, z, n):
        k12_power = mpmath.power(self.k_12, n)
        integral = self.bessel_integral(r, z, n)
        element = k12_power  * integral
        if self.debug:
            self.log(
                "element({:5.2f}, {:>5.2f}, {:d}) = {}".format(r, z, int(n), element)
            )
            self.running_total += element
            self.log("running total = {:>5.2f}".format(float(self.running_total)))
            self.b.append(integral)
            self.c.append(element)
            self.d.append(self.running_total)
            self.e.append(k12_power)
        return element

    def inf_sum_via_bessel(self, r, z):
        result = mpmath.nsum(lambda n: self.element_via_bessel(r, z, n), [1, mpmath.inf])
        return result

    def inf_sum_potential(self, r, z):
        result = mpmath.nsum(lambda n: self.sum_element(r, z, n), [1, mpmath.inf])
        self.log("result = {}".format(result))
        return np.float64(result)

    def field_potential(self, I, r, z):
        q = np.float64(I / (2 * np.pi * self.sigma_1))
        self.running_total = 0
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.e = []
        first = self.one_over_root(r, z, 0)
        second = self.inf_sum_via_bessel(r, z)
        potential = np.float64(q * (first + second))
        self.log(
            "potential ({:10.2f} {:10.2f}) = {:10.2f}".format(r, z, float(potential))
        )
        if np.isinf(potential):
            potential = np.nan
        return potential

    def field_potential_surface(self, I, r):
        return self.field_potential(I, r, 0)

    def over_root_dr(self, r, z, n):
        if np.isclose(np.float64(r), 0, atol=EPS) and np.isclose(
            np.float64(z), 0, atol=EPS
        ):
            return np.inf
        a = np.power(r, 2)
        sz = np.add(z, 2 * n * self.d_1)
        b = np.power(sz, 2)
        if np.isclose(np.float64(a), 0, atol=EPS) and np.isclose(
            np.float64(b), 0, atol=EPS
        ):
            return np.inf
        return r * (np.power(a + b, -3 / 2))

    def inf_sum_strength_dr(self, r, z):
        return mpmath.nsum(
            lambda n: (self.k_12**n)
            * (self.over_root_dr(r, -1 * z, n) + self.over_root_dr(r, z, n)),
            [1, mpmath.inf],
        )

    def over_root_dz(self, r, z, n):
        if np.isclose(np.float64(r), 0, atol=EPS) and np.isclose(
            np.float64(z), 0, atol=EPS
        ):
            return np.inf
        sz = np.add(z, 2 * n * self.d_1)
        a = np.power(r, 2)
        b = np.power(sz, 2)
        if np.isclose(np.float64(a), 0, atol=EPS) and np.isclose(
            np.float64(b), 0, atol=EPS
        ):
            return np.inf
        return sz / (np.power(a + b, 3 / 2))

    def inf_sum_strength_dz(self, r, z):
        return mpmath.nsum(
            lambda n: (self.k_12**n)
            * (-1 * self.over_root_dz(r, -1 * z, n) + self.over_root_dz(r, z, n)),
            [1, mpmath.inf],
        )

    def field_strength(self, I, r, z):
        q = I / (2 * np.pi * self.sigma_1)
        mdVdr = q * (self.over_root_dr(r, z, 0) + self.inf_sum_strength_dr(r, z))
        mdVdz = q * (self.over_root_dz(r, z, 0) + self.inf_sum_strength_dz(r, z))
        return (mdVdr, mdVdz)


if __name__ == "__main__":
    print("Two Layer Model")
    import parameters

    sigma_1 = parameters.sigma_fat
    sigma_2 = parameters.sigma_muscle
    d_1 = parameters.d_1
    r_size = 10 * d_1
    z_size = 10 * d_1
    two_layer_model = TwoLayerModel(sigma_1, sigma_2, d_1)
    r_coords = np.linspace(-r_size, r_size, 11, dtype=np.float64)
    z_coords = np.linspace(0, z_size, 10, dtype=np.float64)
    I = 1 * 10 ** (-3)
    r, z = np.meshgrid(r_coords, z_coords)

    line = []
    v = []
    line_er = []
    er = []
    line_ez = []
    ez = []

    two_layer_model.debug = True
    print("d_1 ", d_1)
    rr = -1.0
    zz = 0.0
    print("rr ", rr)
    print("zz ", zz)
    potential = two_layer_model.field_potential(I, rr, zz)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(two_layer_model.a)
    plt.plot(two_layer_model.b)
    plt.plot(two_layer_model.c)
    plt.plot(two_layer_model.d)
    plt.plot(two_layer_model.e)
    # first = np.ones(len(two_layer_model.d)) * two_layer_model.one_over_root(rr, zz, 0)
    # plt.plot(first)
    # plt.plot(first + two_layer_model.d)
    plt.legend(
        [
            "first root",
            "second root",
            "element",
            "inf sum total",
            "k12^n",
            "simple",
            "sum",
        ]
    )
    plt.show()

    # exit()

    two_layer_model.debug = False

    for zz in z_coords:
        for rr in r_coords:
            potential = two_layer_model.field_potential(I, rr, zz)
            print("potential ({} {}) = {:10.2f}".format(rr, zz, float(potential)))
            # if np.isnan(potential):
            #     potential = line[-1]
            line.append(potential)
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

    parameters = {
        "sigma_1": two_layer_model.sigma_1,
        "sigma_2": two_layer_model.sigma_1,
        "d_1": two_layer_model.d_1,
        "I": I,
    }
    np.save("./computed/{}/parameters.npy".format(two_layer_model.name), parameters)

    np.savetxt("./computed/{}/r.csv".format(two_layer_model.name), r, delimiter=",")
    np.savetxt("./computed/{}/z.csv".format(two_layer_model.name), z, delimiter=",")
