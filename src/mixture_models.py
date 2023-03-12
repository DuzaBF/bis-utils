import scipy.constants as sc

def hanai_equation(sigma, sigma_m, sigma_p):
    return (((sigma - sigma_p) / (sigma_m - sigma_p)) * (((sigma_m) / (sigma)) ** (1/3)))

def cell_sigma(sigma_icf, C_m, t, freq):
    return sigma_icf * (1 + ((2*sigma_icf) / (1j * 2 * sc.pi * freq * C_m * t))) ** (-1)

if __name__ == "__main__":
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