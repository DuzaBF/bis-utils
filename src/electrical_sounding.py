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

def K_12(sigma_1, sigma_2):
    return (sigma_1 - sigma_2) / (sigma_1 - sigma_2)

def Z_single(sigma_1, L, s):
    return (1/sigma_1) * (2/math.pi) * (one_over(L, -1 * s) - one_over(L, s))

def one_over(L, s):
    return 1/(L + s)

def one_over_root(L, s, n, d_1):
    return 1/mpmath.sqrt(((L + s)**2) + ((4 * n * d_1)**2))

def two_layer_factor(sigma_1, sigma_2, L, s, d_1):
    return mpmath.nsum(lambda n: (K_12(sigma_1, sigma_2) ** n) * (one_over_root(L, -1 * s, n, d_1) - one_over_root(L, s, n, d_1)), [1, mpmath.inf])

def Z_two(sigma_1, sigma_2, L, s, d_1):
    return Z_single(sigma_1, L, s) + (1/sigma_1) * (2/math.pi) * (2 * two_layer_factor(sigma_1, sigma_2, L, s, d_1))

def apparent_conductivity(Z, L, s):
    return (1/Z) * (2/math.pi) * (one_over(L, -1 * s) - one_over(L, s))

if __name__ == "__main__":
    L = 30 * 10**(-3) # [m]
    s = 10 * 10**(-3) # [m]
    d_1 = 10 * 10**(-3) # [m]
    sigma_1 = 0.02 # [S/m]
    sigma_2 = 0.34 # [S/m]

    print(Z_single(sigma_1,L, s))
    print(Z_two(sigma_1, sigma_2, L, s, d_1))
    print(apparent_conductivity(Z_two(sigma_1, sigma_2, L, s, d_1), L, s))