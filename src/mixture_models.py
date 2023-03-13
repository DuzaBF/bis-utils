import scipy.constants as sc

def hanai_equation(sigma, sigma_m, sigma_p):
    return (((sigma - sigma_p) / (sigma_m - sigma_p)) * (((sigma_m) / (sigma)) ** (1/3)))

def cell_sigma(sigma_icf, C_m, t, freq):
    return sigma_icf * (1 + ((2*sigma_icf) / (1j * 2 * sc.pi * freq * C_m * t))) ** (-1)

def one_plus_Vi_Ve_var1(sigma_ecf, sigma_tbw, sigma_hf, sigma_lf):
    return ((sigma_ecf * sigma_hf) / (sigma_tbw * sigma_lf)) ** 2/3

def sigma_tbw(sigma_ecf, sigma_icf, sigma_hf, sigma_lf):
    nom = sigma_ecf * sigma_icf
    lf_hf = (sigma_lf/sigma_hf) ** (2/3)
    denom = sigma_ecf - (sigma_ecf - sigma_icf) * lf_hf
    return nom / denom

def one_plus_Vi_Ve_var2(sigma_ecf, sigma_icf, sigma_hf, sigma_lf):
    return (((sigma_ecf * sigma_hf) / (sigma_icf * sigma_lf)) - ((sigma_ecf - sigma_icf) / (sigma_icf))) ** 2/3

def test1():
    import parameters
    import tissue_data as td
    freq = 100000 # [Hz]
    csigma_icf = td.TissueDataComplex.to_complex(freq, parameters.sigma_icf, parameters.eps_icf)
    csigma_ecf = td.TissueDataComplex.to_complex(freq, parameters.sigma_ecf, parameters.eps_ecf)
    csigma_cell = cell_sigma(csigma_icf, parameters.C_m, parameters.t, freq)
    csigma_fat = td.TissueDataComplex.to_complex(freq, parameters.sigma_fat, parameters.eps_fat)
    vol_conc = 1 - hanai_equation(csigma_fat, csigma_ecf, csigma_cell)

    print("{:>32} {:>6}: {:>5.4f} S/m at {:.0f} kHz".format("Complex conductivity of", "a fat", csigma_fat, freq/1000))
    print("{:>32} {:>6}: {:>5.4f} S/m at {:.0f} kHz".format("Complex conductivity of", "an ICF", csigma_icf, freq/1000))
    print("{:>32} {:>6}: {:>5.4f} S/m at {:.0f} kHz".format("Complex conductivity of", "a cell", csigma_cell, freq/1000))
    print("{:>32} {:>6}: {:>5.4f} S/m at {:.0f} kHz".format("Complex conductivity of", "an ECF", csigma_ecf, freq/1000))
    print("{:>40} {:>5.4f}".format("Volume concentration of cells in fat:", vol_conc))

def test2():
    import parameters
    import tissue_data as td
    freq_lf = 1000 # [Hz]
    freq = 100000 # [Hz]
    freq_hf = 1000000 # [Hz]
    csigma_icf = td.TissueDataComplex.to_complex(freq, parameters.sigma_icf, parameters.eps_icf)
    csigma_ecf = td.TissueDataComplex.to_complex(freq, parameters.sigma_ecf, parameters.eps_ecf)
    csigma_fat = td.TissueDataComplex.to_complex(freq, parameters.sigma_fat, parameters.eps_fat)


if __name__ == "__main__":
    # test1()
    pass