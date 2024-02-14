import numpy as np
import matplotlib.pyplot as plt


def plot_one_layer():
    model = "one-layer-model"

    v = np.genfromtxt("./computed/{}/potential.csv".format(model),
                      delimiter=",", usemask=True)
    er = np.genfromtxt(
        "./computed/{}/strength_r.csv".format(model), delimiter=",", usemask=True)
    ez = np.genfromtxt(
        "./computed/{}/strength_z.csv".format(model), delimiter=",", usemask=True)
    parameters = np.genfromtxt(
        "./computed/{}/parameters.csv".format(model), delimiter=",", usemask=True)
    r = np.genfromtxt("./computed/{}/r.csv".format(model),
                      delimiter=",", usemask=True)
    z = np.genfromtxt("./computed/{}/z.csv".format(model),
                      delimiter=",", usemask=True)

    v = np.clip(v, -1, 1)

    norm = np.sqrt(np.add(np.power(er, 2), np.power(ez, 2)))
    er = er / norm
    ez = ez / norm

    figure = plt.figure()

    ax = figure.add_subplot()
    ax.invert_yaxis()
    ax.set_title(
        r"Electric field for one-layer model for I={} A, $\sigma$={} Sm/m".format(parameters[0], parameters[1]))
    ax.set_xlabel("r, m")
    ax.set_ylabel("z, m")

    cset_v = ax.contourf(r, z, v)
    cbi_v = figure.colorbar(cset_v, orientation='horizontal', shrink=0.8)
    cbi_v.set_label('Potential, V')

    n = 8
    q_e = ax.quiver(r[1::n, 1::n], z[1::n, 1::n],
                    er[1::n, 1::n], -ez[1::n, 1::n], norm[1::n, 1::n], pivot="mid", cmap="gist_gray",
                    units="xy",
                    width=0.008,
                    headwidth=2,
                    headlength=3,
                    headaxislength=3,
                    scale_units="xy",
                    scale=10
                    )
    q_e.set_clim(0, 2)
    cbi_e = figure.colorbar(q_e, orientation='horizontal', shrink=0.8)
    cbi_e.set_label('Strength, V/m')

    plt.show()


def plot_two_layer():
    model = "two-layer-model"

    v = np.genfromtxt("./computed/{}/potential.csv".format(model),
                      delimiter=",", usemask=True)
    er = np.genfromtxt(
        "./computed/{}/strength_r.csv".format(model), delimiter=",", usemask=True)
    ez = np.genfromtxt(
        "./computed/{}/strength_z.csv".format(model), delimiter=",", usemask=True)
    parameters = np.genfromtxt(
        "./computed/{}/parameters.csv".format(model), delimiter=",", usemask=True)
    r = np.genfromtxt("./computed/{}/r.csv".format(model),
                      delimiter=",", usemask=True)
    z = np.genfromtxt("./computed/{}/z.csv".format(model),
                      delimiter=",", usemask=True)

    sigma_1 = parameters[0]
    sigma_2 = parameters[1]
    d_1 = parameters[2]
    I = parameters[3]

    v = np.clip(v, -0.5, 0.5)

    er = np.array(er, np.float64)
    ez = np.array(ez, np.float64)
    norm = np.sqrt(np.add(np.power(er, 2), np.power(ez, 2)))
    er = er / norm
    ez = ez / norm

    figure = plt.figure()

    ax = figure.add_subplot()
    ax.invert_yaxis()
    ax.set_title(r"Electric field for two-layer model for I={} A, $\sigma_1$={} Sm/m, $\sigma_2$={} Sm/m, $d_1$={} m".format(I, sigma_1, sigma_2, d_1))
    ax.set_xlabel("r, m")
    ax.set_ylabel("z, m")

    cset_v = ax.contourf(r, z, v)
    cbi_v = figure.colorbar(cset_v, orientation='horizontal', shrink=0.8)
    cbi_v.set_label('Potential, V')
    r_coords = np.linspace(r[0][0], r[0][-1], len(r[0]))
    brd = ax.plot(r_coords, d_1 * np.ones(r_coords.shape), color="black")

    n = 4
    q_e = ax.quiver(r[1::n, 1::n], z[1::n, 1::n],
                    er[1::n, 1::n], -ez[1::n, 1::n], norm[1::n, 1::n], pivot="mid", cmap="gist_gray",
                    units="xy",
                    width=0.008,
                    headwidth=2,
                    headlength=3,
                    headaxislength=3,
                    scale_units="xy",
                    scale=10
                    )
    q_e.set_clim(0, 2)
    cbi_e = figure.colorbar(q_e, orientation='horizontal', shrink=0.8)
    cbi_e.set_label('Strength, V/m')

    plt.show()


if __name__ == "__main__":
    # plot_one_layer()
    plot_two_layer()
