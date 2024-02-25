import two_layer_model
import one_layer_model
import plot_fields
from typing import Union
import numpy as np
import sympy as sym
import scipy.optimize
import mpmath


class Tetrapolar:
    def __init__(self,
                 model: Union[one_layer_model.OneLayerModel, two_layer_model.TwoLayerModel],
                 el_coords: list,
                 ) -> None:
        self.model = model
        self.el_coords = el_coords
        pass

    def impedance(self, current_pair: tuple, measure_pair: tuple) -> np.float64:
        el_A = self.el_coords[current_pair[0]-1]
        el_B = self.el_coords[current_pair[1]-1]
        el_M = self.el_coords[measure_pair[0]-1]
        el_N = self.el_coords[measure_pair[1]-1]

        v_M = self.model.field_potential_surface(
            1, el_M - el_A) + self.model.field_potential_surface(-1, el_M - el_B)
        v_N = self.model.field_potential_surface(
            1, el_N - el_A) + self.model.field_potential_surface(-1, el_N - el_B)

        u_MN = v_N - v_M

        imp = abs(u_MN) / 1

        return imp


def frac(r):
    return 1 / r


def sqrt_frac(r, d):
    return 1 / (sym.sqrt(r**2 + (2*d)**2))


def lsm(xi: list[list], zi: list):
    s_1 = sym.Symbol('sigma_1')
    s_2 = sym.Symbol('sigma_2')
    d_1 = sym.Symbol('d_1')
    x_a = sym.Symbol('x_a')
    x_m = sym.Symbol('x_m')
    x_n = sym.Symbol('x_n')
    x_b = sym.Symbol('x_b')
    n = sym.Symbol('n')
    k12 = (s_1 - s_2) / (s_1 + s_2)
    z = (1 / (2 * sym.pi * s_1)) * (
        frac(sym.Abs(x_m - x_a)) -
        frac(sym.Abs(x_m - x_b)) -
        frac(sym.Abs(x_n - x_a)) +
        frac(sym.Abs(x_n - x_b)) +
        2 * sym.Sum((k12 ** n) * (
            sqrt_frac(x_m - x_a, n*d_1) -
            sqrt_frac(x_m - x_b, n*d_1) -
            sqrt_frac(x_n - x_a, n*d_1) +
            sqrt_frac(x_n - x_b, n*d_1)),
            (n, 1, sym.oo))
    )

    dz_ds1 = sym.diff(z, s_1)

    dz_ds2 = sym.diff(z, s_2)

    dz_dd1 = sym.diff(z, d_1)

    dd_ds1 = 0
    for i in range(3):
        dd_ds1 = dd_ds1 + 2 * (zi[i]-sym.Subs(z, (x_a, x_m, x_n, x_b), xi[i])
                               ) * sym.Subs(dz_ds1, (x_a, x_m, x_n, x_b), xi[i])

    dd_ds2 = 0
    for i in range(3):
        dd_ds2 = dd_ds2 + 2 * (zi[i]-sym.Subs(z, (x_a, x_m, x_n, x_b), xi[i])
                               ) * sym.Subs(dz_ds2, (x_a, x_m, x_n, x_b), xi[i])

    dd_dd1 = 0
    for i in range(3):
        dd_dd1 = dd_dd1 + 2 * (zi[i]-sym.Subs(z, (x_a, x_m, x_n, x_b), xi[i])
                               ) * sym.Subs(dz_dd1, (x_a, x_m, x_n, x_b), xi[i])


def lsm_scipy(xi: list[list], zi: list):
    cnt = len(z)

    def q(s: list) -> np.float64:
        s_1 = s[0]
        return 1 / (2 * mpmath.pi * s_1)

    def g1(x: list) -> np.float64:
        x_a = x[0]
        x_m = x[1]
        x_n = x[2]
        x_b = x[3]
        tmp = (
            frac(abs(x_m - x_a)) -
            frac(abs(x_m - x_b)) -
            frac(abs(x_n - x_a)) +
            frac(abs(x_n - x_b))
        )
        return abs(tmp)

    def g2(x: list, s: list, n) -> np.float64:
        x_a = x[0]
        x_m = x[1]
        x_n = x[2]
        x_b = x[3]
        d_1 = s[2]
        tmp = (
            sqrt_frac(x_m - x_a, d_1 * n) -
            sqrt_frac(x_m - x_b, d_1 * n) -
            sqrt_frac(x_n - x_a, d_1 * n) +
            sqrt_frac(x_n - x_b, d_1 * n)
        )
        return abs(tmp)

    def k12(s: list) -> np.float64:
        s_1 = s[0]
        s_2 = s[1]
        return (s_1 - s_2) / (s_1 + s_2)

    def f(x: list, s: list) -> np.float64:
        return q(s) * (g1(x) + 2 * mpmath.nsum(lambda n: k12(s) ** n * g2(x, s, n), [1, mpmath.inf]))

    def r_i(z, x: list, s: list):
        return z - f(x, s)

    def residual(s: list):
        return np.array([r_i(z[i], xi[i], s) for i in range(cnt)], dtype=np.float64)

    bounds = ([0, 0, 0], [10, 10, 10])

    initial_guesses = [0.1, 0.1, 0.1]
    result = scipy.optimize.least_squares(
        residual, initial_guesses, method='trf', bounds=bounds)
    print(result)

    for i in range(cnt):
        print("z[{}] = {}  f({}, {}) = {}".format(
            i, zi[i], xi[i], result.x, f(xi[i], result.x)))

    return result.x


if __name__ == "__main__":
    x1 = 0
    x2 = 0.1
    x3 = 0.2
    x4 = 0.3
    x_geom = [x1, x2, x3, x4]
    """
    A M N B
    1 2 3 4
    3 1 2 4
    2 1 3 4
    """
    arrangements = [
        ((1, 4), (2, 3)),
        ((3, 4), (1, 2)),
        ((2, 4), (1, 3))
    ]

    xi = []
    for a in arrangements:
        ind_a = a[0][0]-1
        ind_m = a[1][0]-1
        ind_n = a[1][1]-1
        ind_b = a[0][1]-1
        xi.append([
            x_geom[ind_a],
            x_geom[ind_m],
            x_geom[ind_n],
            x_geom[ind_b]
        ])

    s_1 = 1
    s_2 = 2
    d_1 = 0.1
    t = Tetrapolar(two_layer_model.TwoLayerModel(
        s_1, s_2, d_1), x_geom)

    for a in arrangements:
        print("Z_{}{} ".format(a[1][0], a[1][1]), t.impedance(a[0], a[1]))

    z = []
    for a in arrangements:
        z.append(t.impedance(a[0], a[1]))

    result = lsm_scipy(xi, z)

    expected = [s_1, s_2, d_1]
    for i in range(len(result)):
        print("Expected: {} Found: {}".format(expected[i], result[i]))

    t2 = Tetrapolar(two_layer_model.TwoLayerModel(
        result[0], result[1], result[2]), x_geom)
    for a in arrangements:
        print("From estimated Z_{}{} ".format(
            a[1][0], a[1][1]), t2.impedance(a[0], a[1]))
