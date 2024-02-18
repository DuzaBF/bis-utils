import numpy as np
import matplotlib.pyplot as plt


def read_model_fields(model):
    v = np.genfromtxt(
        "./computed/{}/potential.csv".format(model), delimiter=",", usemask=True
    )
    er = np.genfromtxt(
        "./computed/{}/strength_r.csv".format(model), delimiter=",", usemask=True
    )
    ez = np.genfromtxt(
        "./computed/{}/strength_z.csv".format(model), delimiter=",", usemask=True
    )
    parameters = np.genfromtxt(
        "./computed/{}/parameters.csv".format(model), delimiter=",", usemask=True
    )
    r = np.genfromtxt("./computed/{}/r.csv".format(model), delimiter=",", usemask=True)
    z = np.genfromtxt("./computed/{}/z.csv".format(model), delimiter=",", usemask=True)

    return v, er, ez, r, z, parameters


def plot_potential_field(figure, ax, r, z, v):
    cset_v = ax.contourf(r, z, v)
    cbi_v = figure.colorbar(cset_v, orientation="horizontal", shrink=0.8)
    cbi_v.set_label("Potential, V")


def plot_strength_field(figure, ax, r, z, er, ez, n=8):
    norm = np.sqrt(np.add(np.power(er, 2), np.power(ez, 2)))
    er = er / norm
    ez = ez / norm
    q_e = ax.quiver(
        r[1::n, 1::n],
        z[1::n, 1::n],
        er[1::n, 1::n],
        -ez[1::n, 1::n],
        norm[1::n, 1::n],
        pivot="mid",
        cmap="gist_gray",
        units="xy",
        width=0.008,
        headwidth=2,
        headlength=3,
        headaxislength=3,
        scale_units="xy",
        scale=10,
    )
    q_e.set_clim(0, 2)
    cbi_e = figure.colorbar(q_e, orientation="horizontal", shrink=0.8)
    cbi_e.set_label("Strength, V/m")


def plot_one_layer(v, er, ez, r, z, parameters):
    v = np.clip(v, -1, 1)

    figure = plt.figure()

    ax = figure.add_subplot()
    ax.invert_yaxis()
    ax.set_title(
        r"Electric field for one-layer model for I={} A, $\sigma$={} Sm/m".format(
            parameters[0], parameters[1]
        )
    )
    ax.set_xlabel("r, m")
    ax.set_ylabel("z, m")

    plot_potential_field(figure, ax, r, z, v)

    plot_strength_field(figure, ax, r, z, er, ez)


def plot_two_layer(v, er, ez, r, z, parameters):
    sigma_1 = parameters[0]
    sigma_2 = parameters[1]
    d_1 = parameters[2]
    I = parameters[3]

    v = np.clip(v, -0.5, 0.5)

    er = np.array(er, np.float64)
    ez = np.array(ez, np.float64)

    figure = plt.figure()

    ax = figure.add_subplot()
    ax.invert_yaxis()
    ax.set_title(
        r"Electric field for two-layer model for I={} A, $\sigma_1$={} Sm/m, $\sigma_2$={} Sm/m, $d_1$={} m".format(
            I, sigma_1, sigma_2, d_1
        )
    )
    ax.set_xlabel("r, m")
    ax.set_ylabel("z, m")

    plot_potential_field(figure, ax, r, z, v)

    plot_strength_field(figure, ax, r, z, er, ez)

    r_coords = np.linspace(r[0][0], r[0][-1], len(r[0]))
    brd = ax.plot(r_coords, d_1 * np.ones(r_coords.shape), color="black")


def sum_shifted_fileds(field, shift_index):
    return np.concatenate(
        [field, np.zeros((len(field), shift_index))], 1
    ) + np.concatenate([np.zeros((len(field), shift_index)), -field], 1)


def plot_sum_of_two(v, er, ez, r, z, parameters, r_shift):
    sigma_1 = parameters[0]
    sigma_2 = parameters[1]
    d_1 = parameters[2]
    I = parameters[3]

    v = np.clip(v, -1, 1)

    er = np.array(er, np.float64)
    ez = np.array(ez, np.float64)

    r_coords = r[0]
    z_coords = z[:, 0]

    r_step = r_coords[1] - r_coords[0]
    r_shift_index = int(np.ceil(r_shift // r_step))
    r_right = np.linspace(
        r_coords[-1] + r_step,
        r_coords[-1] + (r_shift_index + 1) * r_step,
        r_shift_index,
    )
    r_extended = np.concatenate([r_coords, r_right])
    r_ext, z_ext = np.meshgrid(r_extended, z_coords)

    left_lim = r_extended[0] + r_step * r_shift_index
    rigth_lim = r_extended[-1] - r_step * r_shift_index
    print(r_coords)
    print(left_lim)
    print(rigth_lim)

    v_sum = sum_shifted_fileds(v, r_shift_index)
    er_sum = sum_shifted_fileds(er, r_shift_index)
    ez_sum = sum_shifted_fileds(ez, r_shift_index)

    figure = plt.figure()

    ax = figure.add_subplot()
    ax.invert_yaxis()
    ax.set_title(
        r"Electric field for two-layer model for $\sigma_1$={} Sm/m, $\sigma_2$={} Sm/m, $d_1$={} m and a pair of electrodes with current I={} A and distance between them l = {} m".format(
            sigma_1, sigma_2, d_1, I, r_shift
        )
    )
    ax.set_xlabel("r, m")
    ax.set_ylabel("z, m")

    plot_potential_field(figure, ax, r_ext, z_ext, v_sum)

    plot_strength_field(figure, ax, r_ext, z_ext, er_sum, ez_sum)

    ax.set_xlim(left_lim, rigth_lim)


if __name__ == "__main__":
    model = "one-layer-model"
    v, er, ez, r, z, parameters = read_model_fields(model)
    plot_one_layer(v, er, ez, r, z, parameters)
    model = "two-layer-model"
    v, er, ez, r, z, parameters = read_model_fields(model)
    plot_two_layer(v, er, ez, r, z, parameters)
    plot_sum_of_two(v, er, ez, r, z, parameters, 0.7)

    plt.show()
