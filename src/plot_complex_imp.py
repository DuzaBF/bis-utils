import matplotlib.pyplot as plt
import numpy


def make_figure():
    figure = plt.figure()
    axes = figure.add_axes(rect=[0.1, 0.1, 0.8, 0.8])
    y_pos = 0
    x_pos = 0
    for s in axes.spines:
        axes.spines[s].set(zorder=0.5)
    axes.spines['left'].set_position(('data', y_pos))
    axes.spines['bottom'].set_position(('data', x_pos))
    # Eliminate upper and right axes
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    # Show ticks in the left and lower axes only
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')
    axes.plot(1, axes.spines['bottom'].get_position()[1], ">k",
                        transform=axes.get_yaxis_transform(), clip_on=False)
    axes.plot(axes.spines['left'].get_position()[1], 1, "^k",
                        transform=axes.get_xaxis_transform(), clip_on=False)
    return figure

def plot_imp_data(data=None, figure=None, freq=None, label="data", linestyle="solid", xaxis_name=None, yaxis_name=None, title=None):
    if figure is None:
        figure = plt.figure()
    figure.axes[0].plot(data.real, data.imag, label=label, linestyle=linestyle)
    figure.axes[0].legend()
    if xaxis_name is not None:
        figure.axes[0].set_xlabel(xaxis_name)
    if yaxis_name is not None:
        figure.axes[0].set_ylabel(yaxis_name)
    if title is not None:
        figure.axes[0].set_title(title)

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

    plot_imp_data(data=td_fat.complex_sigma, figure=figure, label="Fat", xaxis_name="Complex conductivity real part, [S/m]", yaxis_name="Complex conductivity imaginary part, [S/m]", title="Complex conductivity")
    # plot_imp_data(data=td_muscle.complex_sigma, figure=figure, label="Muscle")
    plot_imp_data(data=td_apparent.complex_sigma, figure=figure, label="Apparent", linestyle="dashed")

    plt.show()