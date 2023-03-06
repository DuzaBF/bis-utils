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

def plot_imp_data(data=None, figure=None, freq=None, xaxis_name="", yaxis_name="", title=0):
    if figure is None:
        figure = plt.figure()
    figure.axes[0].plot(data.real, data.imag)

if __name__ == "__main__":
    import argparse
    import tissue_data
    import electrical_sounding as es
    parser = argparse.ArgumentParser(
                    prog = 'Plot complex impedance')
    parser.add_argument('-f', '--fat', required=False) 
    parser.add_argument('-m', '--muscle', required=False) 
    args = parser.parse_args()

    td_fat = tissue_data.TissueDataComplex(args.fat)
    td_muscle = tissue_data.TissueDataComplex(args.muscle)

    L = 30 * 10**(-3) # [m]
    s = 10 * 10**(-3) # [m]
    d_1 = 10 * 10**(-3) # [m]

    a = numpy.array(list(zip(td_fat.complex_sigma, td_muscle.complex_sigma)))
    print(a)
    b = list(map(lambda x: es.apparent_conductivity(x[0], x[1], L, s, d_1), zip(td_fat.complex_sigma, td_muscle.complex_sigma)))
    print(b)

    sigma_apparent = numpy.array(list(map(lambda x: es.apparent_conductivity(x[0], x[1], L, s, d_1), zip(td_fat.complex_sigma, td_muscle.complex_sigma))))
    print(td_fat)
    print(td_muscle)
    print(sigma_apparent)

    figure = make_figure()

    plot_imp_data(data=td_fat.complex_sigma, figure=figure)
    plot_imp_data(data=td_muscle.complex_sigma, figure=figure)
    plot_imp_data(data=sigma_apparent, figure=figure)

    plt.show()