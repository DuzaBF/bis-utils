import argparse

def test1():
    import src.electrical_sounding as es
    import src.mixture_models as mm
    import src.parameters as pr

    sigma_ecf = pr.sigma_ecf
    sigma_icf = pr.sigma_icf

    sigma_fat_lf = pr.sigma_fat_lf
    sigma_fat_hf = pr.sigma_fat_hf

    sigma_muscle_lf = pr.sigma_muscle_lf
    sigma_muscle_hf = pr.sigma_muscle_hf

    L = pr.L
    s = pr.s
    d_1 = pr.d_1

    sigma_a_lf = es.apparent_conductivity(sigma_fat_lf, es.single_layer_geom_coef(L, s), es.two_layer_geom_coef(es.K_12(sigma_fat_lf, sigma_muscle_lf), L, s, d_1)).real
    sigma_a_hf = es.apparent_conductivity(sigma_fat_hf, es.single_layer_geom_coef(L, s), es.two_layer_geom_coef(es.K_12(sigma_fat_hf, sigma_muscle_hf), L, s, d_1)).real

    Z1_lf = es.impedance(sigma_fat_lf, es.single_layer_geom_coef(L, s))
    Z1_hf = es.impedance(sigma_fat_hf, es.single_layer_geom_coef(L, s))
    Z2_lf = float(es.impedance(sigma_fat_lf, es.two_layer_geom_coef(es.K_12(sigma_fat_lf, sigma_muscle_lf), L, s, d_1)))
    Z2_hf = float(es.impedance(sigma_fat_hf, es.two_layer_geom_coef(es.K_12(sigma_fat_hf, sigma_muscle_hf), L, s, d_1)))

    vi_ve_true = mm.one_plus_Vi_Ve_var3(sigma_ecf, sigma_icf, sigma_fat_hf, sigma_fat_lf) - 1
    vi_ve_ap = mm.one_plus_Vi_Ve_var3(sigma_ecf, sigma_icf, sigma_a_hf, sigma_a_lf) - 1

    err  = abs(vi_ve_true - vi_ve_ap) / vi_ve_true

    print("{}={} mm, {}={} mm, {}={} mm".format("L", L*1000, "s", s*1000, "d_1", d_1*1000))
    print("{}={} S/m, {}={} S/m".format("sigma_ecf", sigma_ecf, "sigma_icf", sigma_icf))
    print("{}={} S/m, {}={} S/m".format("sigma_fat_lf", sigma_fat_lf, "sigma_fat_hf", sigma_fat_hf))
    print("{}={} S/m, {}={} S/m".format("sigma_muscle_lf", sigma_muscle_lf, "sigma_muscle_hf", sigma_muscle_hf))

    print("{}: {:.4f} S/m".format("Apparent LF conductivity", sigma_a_lf))
    print("{}: {:.4f} S/m".format("Apparent HF conductivity", sigma_a_hf))
    
    print("{}: {:.4f}".format("True volume ratio of ICF to ECF", vi_ve_true))
    print("{}: {:.4f}".format("Apparent volume ratio of ICF to ECF", vi_ve_ap))

    print("{}: {:.1f} %".format("Relative error", err * 100))

    print("Measured impedance:")
    print("{} = {:.4f} Ohm".format("Z1_lf", Z1_lf))
    print("{} = {:.4f} Ohm".format("Z1_hf", Z1_hf))
    print("{} = {:.4f} Ohm".format("Z2_lf", Z2_lf))
    print("{} = {:.4f} Ohm".format("Z2_hf", Z2_hf))

def test2():
    import src.electrical_sounding as es
    import src.parameters as pr

    sigma_1 = pr.sigma_fat_lf

    sigma_2 = pr.sigma_muscle_lf

    L = pr.L
    s = pr.s
    d_1 = pr.d_1

    sigma_a_lf = es.apparent_conductivity(sigma_1, es.single_layer_geom_coef(L, s), es.two_layer_geom_coef(es.K_12(sigma_1, sigma_2), L, s, d_1)).real

    Z1 = es.impedance(sigma_1, es.single_layer_geom_coef(L, s))
    Z2 = es.impedance(sigma_2, es.single_layer_geom_coef(L, s))
    Z12 = float(es.impedance(sigma_1, es.two_layer_geom_coef(es.K_12(sigma_1, sigma_2), L, s, d_1)))

    print("{}={} mm, {}={} mm, {}={} mm".format("L", L*1000, "s", s*1000, "d_1", d_1*1000))
    print("{}={} S/m".format("sigma_1", sigma_1))
    print("{}={} S/m".format("sigma_2", sigma_2))

    print("{}: {:.4f} S/m".format("Apparent conductivity", sigma_a_lf))

    print("Measured impedance:")
    print("{} = {:.4f} Ohm".format("Z1", Z1))
    print("{} = {:.4f} Ohm".format("Z2", Z2))
    print("{} = {:.4f} Ohm".format("Z12", Z12))

def main():
    parser = argparse.ArgumentParser(
                prog = 'BIS utilities')
    parser.add_argument('filename') 
    args = parser.parse_args()

if __name__ == "__main__":
    test2()
