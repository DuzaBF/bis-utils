'''
Symmetrical tetrapolar electrode system:
L - distance between current electrodes
s - distance between measurement electrodes
d_1 - thickness of fat layer
sigma_fat - conductivity of fat at 100 kHz
eps_fat - dielectric constant of fat at 100 kHz
sigma_muscle - conductivity of muscle at 100 kHz
eps_muscle - dielectric constant of muscle at 100 kHz
eps_icf - dielectric constant of the intracellular fluid (cytoplasm)
sigma_icf - conductivity of the intracellular fluid (cytoplasm)
eps_ecf - dielectric constant of the extracellular fluid
sigma_ecf - conductivity of the extracellular fluid
C_m - cell membrane capacitance per unit area
t - size of the cell
'''
L = 25 * 10**(-3) # [m]
s = 10 * 10**(-3) # [m]
d_1 = 10 * 10**(-3) # [m]
sigma_fat = 0.02 # [S/m]
eps_fat = 92.885
sigma_muscle = 0.34 # [S/m]
eps_muscle = 8089.2
sigma_muscle_along = 0.44 # [S/m]
eps_muscle_along = 10000
sigma_muscle_across = 0.12 # [S/m]
eps_muscle_across = 3000
eps_icf = 80
sigma_icf = 0.3 # [S/m]
eps_ecf = 80
sigma_ecf = 1.2 # [S/m]
C_m = 0.01 # [F/m^2]
t = 30 * 10**(-6) # [m]

sigma_fat_lf = 0.022 # [S/m]
sigma_fat_hf = 0.025 # [S/m]

sigma_muscle_lf = 0.321 # [S/m]
sigma_muscle_hf = 0.503 # [S/m]
