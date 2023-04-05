import sympy as sym

if __name__ == "__main__":
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    print(sym.expand((x + y) ** 3))