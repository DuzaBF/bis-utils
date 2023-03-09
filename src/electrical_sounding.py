import math
import mpmath

'''
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
    return complex("{}{}{}j".format(sigma_a.real, ["+","-"][sigma_a.imag < 0], sigma_a.imag))

def impedance(sigma, geom_coef):
    return geom_coef / sigma

if __name__ == "__main__":
    L = 30 * 10**(-3) # [m]
    s = 10 * 10**(-3) # [m]
    d_1 = 10 * 10**(-3) # [m]
    sigma_1 = 0.02 # [S/m]
    sigma_2 = 0.34 # [S/m]

    print("{:>40} {:>10} {} [Ohm]".format("Impedance of a single fat layer:", " Z =", impedance(sigma_1, single_layer_geom_coef(L, s))))
    print("{:>40} {:>10} {} [Ohm]".format("Impedance of a fat and muscle layers:", "Z =", impedance(sigma_1, two_layer_geom_coef(K_12(sigma_1, sigma_2), L, s, d_1))))
    print("{:>40} {:>10} {} [S/m]".format("Apparent conductivity:", "sigma_a =", apparent_conductivity(sigma_1, single_layer_geom_coef(L, s), two_layer_geom_coef(K_12(sigma_1, sigma_2), L, s, d_1))))