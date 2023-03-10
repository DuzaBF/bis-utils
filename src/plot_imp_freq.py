import matplotlib.pyplot as plt
import numpy


def make_figure():
    figure = plt.figure()
    axes = figure.add_axes(rect=[0.1, 0.1, 0.8, 0.8])
    return figure

def plot_imp_data(freq=None, data=None, figure=None, xaxis_name="", yaxis_name="", title=0):
    if figure is None:
        figure = plt.figure()
    figure.axes[0].plot(freq, numpy.abs(data))
    figure.axes[0].set_xscale("log")

if __name__ == "__main__":
    import argparse
    import tissue_data as td
    import electrical_sounding as es
    import parameters
    parser = argparse.ArgumentParser(
                    prog = 'Plot complex conductivity')
    parser.add_argument('-f', '--fat', required=False) 
    parser.add_argument('-m', '--muscle', required=False) 
    args = parser.parse_args()

    tdf_fat = td.TabularDataFile(args.fat)
    tdf_muscle = td.TabularDataFile(args.muscle)

    td_fat = td.TissueDataComplex(freq=tdf_fat.data.columns[0], sigma_real=tdf_fat.data.columns[1], eps=tdf_fat.data.columns[2], name=args.fat)
    td_muscle = td.TissueDataComplex(freq=tdf_muscle.data.columns[0], sigma_real=tdf_muscle.data.columns[1], eps=tdf_muscle.data.columns[2], name=args.muscle)

    L = parameters.L
    s = parameters.s
    d_1 = parameters.d_1

    geom_coef_1 = es.single_layer_geom_coef(L, s)
    k_12_f = numpy.fromiter(map(lambda x: es.K_12(x[0], x[1]), zip(td_fat.complex_sigma, td_muscle.complex_sigma)), dtype=numpy.csingle)
    geom_coef_2 = numpy.fromiter(map(lambda x : es.two_layer_geom_coef(x, L, s, d_1), k_12_f), dtype=numpy.csingle)

    sigma_apparent = numpy.fromiter(map(lambda x: es.apparent_conductivity(x[0], geom_coef_1, x[1]), zip(td_fat.complex_sigma, geom_coef_2)), dtype=numpy.csingle)
    td_apparent = td.TissueDataComplex(freq=td_fat.freq, complex_sigma=sigma_apparent, name="apparent")
    print(td_fat)
    print(td_muscle)
    print(td_apparent)

    figure = make_figure()

    plot_imp_data(freq=td_fat.freq, data=td_fat.complex_sigma, figure=figure)
    plot_imp_data(freq=td_fat.freq, data=td_muscle.complex_sigma, figure=figure)
    plot_imp_data(freq=td_fat.freq, data=sigma_apparent, figure=figure)

    plt.show()