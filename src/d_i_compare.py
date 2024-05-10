import scipy.constants
import numpy

"""
This module contains a function, which compares conductivity and displacement currents for a given media and frequency.
For reference, see:
L. D. Landau, E. M. Lifšic, L. P. Pitaevskij, and L. D. Landau, “The field equations in a dielectric in the absence of dispersion,” in Electrodynamics of continuous media, 2. ed., rev.Enlarged, Repr., Amsterdam Heidelberg: Elsevier Butterworth-Heinemann, 2009.
"""

def calc_ratio(eps, sigma, omega):
    return (eps * scipy.constants.epsilon_0 * omega) / (sigma)

"""
Skin depth
The magnetic permeability of a biological tissue is close to that of the free space μ0 and assumed to be the constant.
The quasi-static approximation is valid for bodies of moderate conductivity, 
if the dimentions of interest are much smaller than the electromagnetic wavelength and the skin depth. 
The skin depth characterizes the outer layer of a conductor where most
electric current flows; it is is most pronounced with very conductive materials in which significant eddy currents flow
Reference:
D. Holder and A. Adler, Eds., Electrical impedance tomography: methods, history and applications, Second. Boca Raton: CRC Press, 2022.
"""
def skin_depth(freq, mu):
    return 1 / numpy.sqrt(scipy.constants.pi * freq * scipy.constants.mu_0 * mu * sigma)

def wavelength(freq, eps, mu):
    c = 1 / numpy.sqrt(eps * scipy.constants.epsilon_0  * scipy.constants.mu_0 * mu)
    return c / freq

if __name__ == "__main__":
    """
    For example the properties of adipose tissue at 100 kHz are taken.
    sigma = 0.024414 [S/m]
    eps = 92.885
    References:
    D. Andreuccetti, R. Fossi, and C. Petrucci, “An Internet resource for the calculation of the dielectric properties of body tissues in the frequency range 10 Hz - 100 GHz,” 1997. http://niremf.ifac.cnr.it/tissprop
    S. Gabriel, R. W. Lau, and C. Gabriel, “The dielectric properties of biological tissues: II. Measurements in the frequency range 10 Hz to 20 GHz,” Phys. Med. Biol., vol. 41, no. 11, pp. 2251–2269, Nov. 1996, doi: 10.1088/0031-9155/41/11/002.
    """
    sigmas = [0.024414, 0.36185, 0.44, 0.12]
    epss = [
        92.885,
        8089.2,
        10000,
        3000,
    ]
    mus = [1, 1, 1, 1]
    freqs = [100 * 1000, 100 * 1000, 100 * 1000, 100 * 1000]
    names = ["Adipose", "Muscle (Mixed)", "Muscle (Along)", "Muscle (Across)"]
    table_row_names = ("Name", "f [kHz]", "sigma [S/m]", "eps", "Id/Ic", "delta, m", "Wavelength, m")
    table_row_template = ""
    for i in range(len(table_row_names)):
        table_row_template += "{:>15s} | "
    print(table_row_template.format(*table_row_names))
    for i in range(4):
        name = names[i]
        sigma = sigmas[i]
        eps = epss[i]
        freq = freqs[i]
        ratio = calc_ratio(eps, sigma, 2 * scipy.constants.pi * freq)
        delta = skin_depth(freq, mus[i])
        wl_lambda = wavelength(freq, eps, mus[i])
        values = ["{:5.5f}".format(x) for x in [freq / 1000, sigma, eps, ratio, delta, wl_lambda]]
        print(table_row_template.format(name, *values))
