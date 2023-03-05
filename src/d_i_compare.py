import scipy.constants

'''
This module contains a function, which compares conductivity and displacement currents for a given media and frequency.
For reference, see:
L. D. Landau, E. M. Lifšic, L. P. Pitaevskij, and L. D. Landau, “The field equations in a dielectric in the absence of dispersion,” in Electrodynamics of continuous media, 2. ed., rev.Enlarged, Repr., Amsterdam Heidelberg: Elsevier Butterworth-Heinemann, 2009.
'''

def calc_ratio(eps, sigma, omega):
    return (eps * scipy.constants.epsilon_0 * omega) / (4 * scipy.constants .pi * sigma)

if __name__ == "__main__":
    '''
    For example the properties of adipose tissue at 100 kHz are taken.
    sigma = 0.024414 [S/m]
    eps = 92.885
    Reference: D. Andreuccetti, R. Fossi, and C. Petrucci, “An Internet resource for the calculation of the dielectric properties of body tissues in the frequency range 10 Hz - 100 GHz,” 1997. http://niremf.ifac.cnr.it/tissprop
    '''
    sigma = 0.024414
    eps = 92.885
    freq = 100 * 1000
    print("Adipose tissue:\nFrequency freq = %s [Hz]\nConductivity sigma = %s [S/m]\nRelative permittivity: eps = %s" % (freq, sigma, eps))
    ratio = calc_ratio(eps, sigma, 2 * scipy.constants.pi * freq)
    print("Conductive and Displacement current comparison ratio eps * omega / (4 * pi * sigma): %s" % ratio)
    print("Ratio small, body behaves as regular conductor") if ratio < 1 else print("Ratio big, body behaves as dielectric")