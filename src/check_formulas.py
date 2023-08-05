import sympy as sym

def matthie_deriv():
    sym.init_printing()
    sigma_lf = sym.Symbol("S_LF", positive=True)
    sigma_hf = sym.Symbol("S_HF", positive=True)
    sigma_ecf = sym.Symbol("S_ECF", positive=True)
    sigma_icf = sym.Symbol("S_ICF", positive=True)
    sigma_tbw = sym.Symbol("S_TBW", positive=True)
    v_ecf = sym.Symbol("V_ECF", positive=True)
    v_icf = sym.Symbol("V_ICF", positive=True)
    v_tbw = sym.Symbol("V_TBW", positive=True)
    v_tot = sym.Symbol("V_tot", positive=True)
    phi_lf = sym.Symbol("P_LF", positive=True)
    phi_hf = sym.Symbol("P_HF", positive=True)
    vvvv = sym.Symbol("1+V_ICF/V_ECF", positive=True)
    slfhf = sym.Symbol("(S_LF/S_HF)", positive=True)
    shflf = sym.Symbol("(S_HF/S_LF)", positive=True)

    hanai_lf_eq = sym.Eq((sigma_lf / sigma_ecf), (1-phi_lf)**sym.Rational(3,2))
    print("Hanai equation for low frequencies")
    sym.pprint(hanai_lf_eq)
    hanai_lf_vol = 1 - v_ecf/v_tot
    hanai_lf_vol_eq = hanai_lf_eq.subs(phi_lf, hanai_lf_vol)

    hanai_hf_eq = sym.Eq((sigma_hf / sigma_tbw), (1-phi_hf)**sym.Rational(3,2))
    print("Hanai equation for high frequencies")
    sym.pprint(hanai_hf_eq)
    hanai_hf_vol = 1 - (v_ecf + v_icf)/v_tot
    hanai_hf_vol_eq = hanai_hf_eq.subs(phi_hf, hanai_hf_vol)

    print("Volume expression for low frequencies")
    sym.pprint(hanai_lf_vol_eq)
    print("Volume expression for high frequencies")
    sym.pprint(hanai_hf_vol_eq)

    v_tot_expr_2 = sym.solve(hanai_hf_vol_eq, v_tot)[0]
    vol_expr_t = hanai_lf_vol_eq.subs(v_tot, v_tot_expr_2)
    vol_expr = sym.Eq(1 + v_icf/v_ecf, ((sigma_ecf * sigma_hf)/(sigma_tbw * sigma_lf))**sym.Rational(2,3))
    vol_expr = vol_expr.subs(sigma_lf, slfhf * sigma_hf)
    vol_expr2 = sym.Eq(1 + v_icf/v_ecf, ((sigma_ecf / sigma_tbw)*(1 / slfhf))**sym.Rational(2,3))
    print("Volume expression from conductivities")
    sym.pprint(vol_expr_t.simplify())
    sym.pprint(vol_expr.simplify())
    sym.pprint(vol_expr2.simplify())

    matthie_eq = sym.Eq((((sigma_tbw - sigma_icf)/(sigma_ecf - sigma_icf)) * (sigma_ecf/sigma_tbw)**sym.Rational(1, 3))**sym.Rational(-1, 1), (v_ecf/(v_icf+v_ecf))**sym.Rational(-1, 1))
    print("Matthie equation for TBW")
    sym.pprint(matthie_eq)
    matthie_eq_sig = matthie_eq = sym.Eq((((sigma_tbw - sigma_icf)/(sigma_ecf - sigma_icf)) * (sigma_ecf/sigma_tbw)**sym.Rational(1, 3))**sym.Rational(-1, 1), ((sigma_ecf * sigma_hf)/(sigma_tbw * sigma_lf))**sym.Rational(2,3))
    sigma_tbw_sol = sym.simplify(sym.solve(matthie_eq_sig, sigma_tbw)[0])
    sigma_tbw_sol = sigma_tbw_sol.subs(sigma_lf, slfhf * sigma_hf).factor()
    print("Matthie sigma TBW")
    sigma_tbw_expr = sym.Eq(sigma_tbw, sigma_tbw_sol)
    sym.pprint(sigma_tbw_expr.simplify())

    matthie_fin = vol_expr.subs(sigma_tbw, sigma_tbw_sol)
    matthie_fin = matthie_fin.subs(slfhf, 1/shflf)
    print("Final Matthie equation")
    simple_matthie = matthie_fin.rhs**sym.Rational(3,2)
    simple_matthie = simple_matthie.expand().collect(shflf).subs(shflf, sigma_hf/sigma_lf)
    matthie_fin2 = sym.Eq(matthie_fin.lhs, simple_matthie**sym.Rational(2,3))
    sym.pprint(matthie_fin2)


def other():
    sym.init_printing()

    csigma = sym.Symbol('sigma')
    csigma_p = sym.Symbol('sigma_p')
    csigma_m = sym.Symbol('sigma_m')
    phi = sym.Symbol('Phi', positive=True)
    hanai_left = ((csigma - csigma_p) / (csigma_m - csigma_p)) * (((csigma_m) / (csigma))**sym.Rational(1,3))
    hanai_expr = sym.Eq(hanai_left, 1 - phi)
    sym.pprint(hanai_expr)

    sigma_lf = sym.Symbol('sigma_lf', positive=True)
    sigma_ecf = sym.Symbol('sigma_ecf', positive=True)
    phi_lf = sym.Symbol('Phi_lf', positive=True)

    sigma_hf = sym.Symbol('sigma_hf', positive=True)
    sigma_tbw = sym.Symbol('sigma_tbw', positive=True)
    phi_hf = sym.Symbol('Phi_hf', positive=True)

    sigma_icf = sym.Symbol('sigma_icf', positive=True)

    v_icf = sym.Symbol('V_icf', positive=True)
    v_ecf = sym.Symbol('V_ecf', positive=True)
    v_tbw = sym.Symbol('V_tbw', positive=True)
    v_tot = sym.Symbol('V_tot', positive=True)

    hanai_lf = hanai_expr.subs([(csigma, sigma_lf), (csigma_p, 0), (csigma_m, sigma_ecf), (phi, phi_lf)])
    sym.pprint(hanai_lf)

    hanai_hf = hanai_expr.subs([(csigma, sigma_hf), (csigma_p, 0), (csigma_m, sigma_tbw), (phi, phi_hf)])
    sym.pprint(hanai_hf)

    phi_lf_expr = 1 - (v_ecf) / (v_tot)
    phi_hf_expr = 1 - (v_ecf + v_icf) / (v_tot)

    hanai_lf_vol = hanai_lf.subs(phi_lf, phi_lf_expr)
    sym.pprint(hanai_lf_vol)

    hanai_hf_vol = hanai_hf.subs(phi_hf, phi_hf_expr)
    sym.pprint(hanai_hf_vol)

    v_tot_expr_1 = sym.Eq(v_tot, sym.solve(hanai_lf_vol, v_tot)[0])
    sym.pprint(v_tot_expr_1)
    v_tot_expr_2 = sym.solve(hanai_hf_vol, v_tot)[0]
    sym.pprint(v_tot_expr_2)

    expr_1 = v_tot_expr_1.subs(v_tot, v_tot_expr_2)
    sym.pprint(expr_1)

    expr_2 = sym.solve(expr_1, v_icf)[0]
    sym.pprint(expr_2)

    one_vi_ve_ex = 1 + v_icf/v_ecf
    one_vi_ve_ex_2 = sym.simplify(one_vi_ve_ex.subs(v_icf, expr_2))
    one_vi_ve_ex_3 = sym.Eq(one_vi_ve_ex, one_vi_ve_ex_2)
    sym.pprint(sym.simplify(one_vi_ve_ex_3))

    tbw_expr = sym.Eq(hanai_left.subs(
        [(csigma, sigma_tbw), (csigma_p, sigma_icf), (csigma_m, sigma_ecf)]
    ), 1 / one_vi_ve_ex_2)
    sym.pprint(sym.simplify(tbw_expr))

    tbw_sol = sym.solve(tbw_expr, sigma_tbw)[0]
    sym.pprint(tbw_sol)

    fin_expr = one_vi_ve_ex_2.subs(sigma_tbw, tbw_sol)
    sym.pprint(sym.expand(fin_expr))

if __name__ == "__main__":
    matthie_deriv()