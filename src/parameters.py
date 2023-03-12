'''
Symmetrical tetrapolar electrode system:
L - distance between current electrodes
s - distance between measurement electrodes
d_1 - thickness of fat layer
sigma_fat - conductivity of fat at 100 kHz
sigma_muscle - conductivity of muscle at 100 kHz
eps_icf - dielectric constant of the intracellular fluid (cytoplasm)
sigma_icf - conductivity of the intracellular fluid (cytoplasm)
sigma_ecf - conductivity of the extracellular fluid
C_m - cell membrane capacitance per unit length
t - size of the cell
'''
L = 60 * 10**(-3) # [m]
s = 40 * 10**(-3) # [m]
d_1 = 5 * 10**(-3) # [m]
sigma_fat = 0.02 # [S/m]
sigma_muscle = 0.34 # [S/m]
eps_icf = 80
sigma_icf = 0.3 # [S/m]
sigma_exf = 1.2 # [S/m]
C_m = 0.01 # [F/m^2]
t = 30 * 10**(-6) # [m]