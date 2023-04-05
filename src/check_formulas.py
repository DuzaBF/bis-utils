import sympy as sym

if __name__ == "__main__":
    sym.init_printing()

    csigma = sym.Symbol('sigma')
    csigma_p = sym.Symbol('sigma_p')
    csigma_m = sym.Symbol('sigma_m')
    phi = sym.Symbol('Phi', positive=True)
    hanai_expr = sym.Eq(((csigma - csigma_p) / (csigma_m - csigma_p)) * (((csigma_m) / (csigma))**sym.Rational(1,3)), 1 - phi)
    sym.pprint(hanai_expr)

    sigma_lf = sym.Symbol('sigma_lf', positive=True)
    sigma_ecf = sym.Symbol('sigma_ecf', positive=True)
    phi_lf = sym.Symbol('Phi_lf', positive=True)
    hanai_lf = hanai_expr.subs([(csigma, sigma_lf), (csigma_p, 0), (csigma_m, sigma_ecf), (phi, phi_lf)])
    sym.pprint(hanai_lf)

    sigma_hf = sym.Symbol('sigma_hf', positive=True)
    sigma_tbw = sym.Symbol('sigma_tbw', positive=True)
    phi_hf = sym.Symbol('Phi_hf', positive=True)
    hanai_hf = hanai_expr.subs([(csigma, sigma_hf), (csigma_p, 0), (csigma_m, sigma_tbw), (phi, phi_hf)])
    sym.pprint(hanai_hf)

    v_icf = sym.Symbol('V_icf', positive=True)
    v_ecf = sym.Symbol('V_ecf', positive=True)
    v_tbw = sym.Symbol('V_tbw', positive=True)
    v_tot = sym.Symbol('V_tot', positive=True)

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