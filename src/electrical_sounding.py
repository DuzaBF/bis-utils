import math
import mpmath
import two_layer_model
import one_layer_model
from typing import Union


'''
AB = L
MN = s

      A  M   N  B
------*--*---*--*-------------------------
| d1                  sigma1(f), eps1(f) |
------------------------------------------
|                     sigma2(f), eps2(f) |
|                                        |
'''


def one_over(L, s):
    return 1/(L + s)


def one_over_root(L, s, n, d_1):
    return 1/mpmath.sqrt(((L + s)**2) + ((4 * n * d_1)**2))


def single_layer_geom_coef(L, s):
    return (2/math.pi) * (one_over(L, -1 * s) - one_over(L, s))


def K_12(sigma_1, sigma_2):
    return (sigma_1 - sigma_2) / (sigma_1 + sigma_2)


def two_layer_factor(K_12, L, s, d_1):
    return mpmath.nsum(lambda n: (K_12 ** n) * (one_over_root(L, -1 * s, n, d_1) - one_over_root(L, s, n, d_1)), [1, mpmath.inf])


def two_layer_geom_coef(K_12, L, s, d_1):
    return single_layer_geom_coef(L, s) + (2/math.pi) * (2 * two_layer_factor(K_12, L, s, d_1))


def apparent_conductivity(sigma_1, geom_coef_1, geom_coef_2):
    sigma_a = sigma_1 * geom_coef_1 / geom_coef_2
    return complex("{}{}{}j".format(sigma_a.real, ["+", "-"][sigma_a.imag < 0], sigma_a.imag))


def impedance(sigma, geom_coef):
    return geom_coef / sigma


if __name__ == "__main__":
    import parameters
    L = parameters.L
    s = parameters.s
    d_1 = parameters.d_1
    sigma_1 = parameters.sigma_fat
    sigma_2 = parameters.sigma_muscle

    print("{:>40} {:>10} {} [Ohm]".format(
        "Impedance of a single fat layer:", " Z =", impedance(sigma_1, single_layer_geom_coef(L, s))))
    print("{:>40} {:>10} {} [Ohm]".format("Impedance of a fat and muscle layers:", "Z =", impedance(
        sigma_1, two_layer_geom_coef(K_12(sigma_1, sigma_2), L, s, d_1))))
    print("{:>40} {:>10} {} [S/m]".format("Apparent conductivity:", "sigma_a =", apparent_conductivity(
        sigma_1, single_layer_geom_coef(L, s), two_layer_geom_coef(K_12(sigma_1, sigma_2), L, s, d_1))))

    def model_impedance(model: Union[one_layer_model.OneLayerModel, two_layer_model.TwoLayerModel], x_a, x_b, x_m, x_n):
        I = 1
        v_m = model.field_potential_surface(I, x_m - x_a) + \
            model.field_potential_surface(-I, x_m - x_b)

        v_n = model.field_potential_surface(I, x_n - x_a) + \
            model.field_potential_surface(-I, x_n - x_b)

        z = abs(v_n - v_m)/I

        return z

    def geom_imp(model: Union[one_layer_model.OneLayerModel, two_layer_model.TwoLayerModel], x_a, x_b, x_m, x_n):
        g_m = model.geom_coef(abs(x_m - x_a), 0) - \
            model.geom_coef(abs(x_m - x_b), 0)

        g_n = model.geom_coef(abs(x_n - x_a), 0) - \
            model.geom_coef(abs(x_n - x_b), 0)

        return abs(g_n - g_m)

    x_a = 0
    x_b = L
    x_m = (L-s)/2
    x_n = (L+s)/2

    one_l = one_layer_model.OneLayerModel(sigma_1)
    two_l = two_layer_model.TwoLayerModel(sigma_1, sigma_2, d_1)

    imp_1 = model_impedance(one_l, x_a, x_b, x_m, x_n)
    imp_2 = model_impedance(two_l, x_a, x_b, x_m, x_n)
    print(geom_imp(one_l, x_a, x_b, x_m, x_n) / (2 * math.pi * one_l.sigma))
    ap = geom_imp(one_l, x_a, x_b, x_m, x_n) / (2 * math.pi * imp_2)

    print("{:>40} {:>10} {} [Ohm]".format(
        "Impedance of a single fat layer:", " Z =", imp_1))
    print("{:>40} {:>10} {} [Ohm]".format(
        "Impedance of a fat and muscle layers:", "Z =", imp_2))
    print(
        "{:>40} {:>10} {} [S/m]".format("Apparent conductivity:", "sigma_a =", ap))
