import matplotlib.pyplot as plt
import numpy


def plot_imp_data(data, freq=None, xaxis_name="", yaxis_name="", title=0):
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
    axes.scatter(data.real, data.imag, facecolor="black")
    axes.plot(1, axes.spines['bottom'].get_position()[1], ">k",
                        transform=axes.get_yaxis_transform(), clip_on=False)
    axes.plot(axes.spines['left'].get_position()[1], 1, "^k",
                        transform=axes.get_xaxis_transform(), clip_on=False)

    plt.show()


if __name__ == "__main__":
    import argparse
    import tissue_data
    parser = argparse.ArgumentParser(
                    prog = 'Plot complex impedance')
    parser.add_argument('filename') 
    args = parser.parse_args()

    td = tissue_data.TissueData(args.filename)

    print(td.complex_sigma)

    plot_imp_data(td.complex_sigma)
