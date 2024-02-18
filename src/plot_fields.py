import numpy as np
import matplotlib.pyplot as plt


class ElectrodeFields:

    def __init__(self, v, er, ez, r, z):
        v = np.array(v, np.float64)
        er = np.array(er, np.float64)
        ez = np.array(ez, np.float64)
        self.v = v
        self.er = er
        self.ez = ez
        self.r = r
        self.z = z

    @property
    def r_step(self):
        return self.r[0][1] - self.r[0][0]

    @property
    def z_step(self):
        return self.z[0][1] - self.z[0][0]

    @property
    def r_coords(self):
        return self.r[0]

    @property
    def z_coords(self):
        return self.z[:, 0]

    @staticmethod
    def extend_field_right(field, r_shift_index):
        return np.concatenate([field, np.zeros((len(field), r_shift_index))], 1)

    @staticmethod
    def extend_field_left(field, r_shift_index):
        return np.concatenate([np.zeros((len(field), r_shift_index)), field], 1)

    @staticmethod
    def sum_shifted_fields(field_1, field_2, r_shift_index):
        return ElectrodeFields.extend_field_right(
            field_1, r_shift_index
        ) + ElectrodeFields.extend_field_left(field_2, r_shift_index)

    def add(self, other, r_shift):
        if type(other) != type(self):
            raise TypeError("Can only add two ElectrodeFields")
        if (self.r_step != other.r_step) or (self.z_step != other.z_step):
            raise ValueError("Fields must be on the same grid")
        if r_shift < 0:
            raise ValueError("Use non-negative distance between sources")

        r_shift_index = int(np.ceil(r_shift // self.r_step))

        r_right = np.linspace(
            self.r_coords[-1] + self.r_step,
            self.r_coords[-1] + (r_shift_index + 1) * self.r_step,
            r_shift_index,
        )
        r_extended = np.concatenate([self.r_coords, r_right])
        r_ext, z_ext = np.meshgrid(r_extended, self.z_coords)

        v_sum = self.sum_shifted_fields(self.v, other.v, r_shift_index)
        er_sum = self.sum_shifted_fields(self.er, other.er, r_shift_index)
        ez_sum = self.sum_shifted_fields(self.ez, other.ez, r_shift_index)

        r_cut = r_ext[:, r_shift_index:-r_shift_index]
        z_cut = z_ext[:, r_shift_index:-r_shift_index]
        v_cut = v_sum[:, r_shift_index:-r_shift_index]
        er_cut = er_sum[:, r_shift_index:-r_shift_index]
        ez_cut = ez_sum[:, r_shift_index:-r_shift_index]

        return ElectrodeFields(v_cut, er_cut, ez_cut, r_cut, z_cut)

    def dot_product(self, other):
        if type(other) != type(self):
            raise TypeError("Can only multiply two ElectrodeFields")
        if (self.r_step != other.r_step) or (self.z_step != other.z_step):
            raise ValueError("Fields must be on the same grid")

        return self.er * other.er + self.ez * other.ez


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


def plot_scalar_field(figure: plt.Figure, ax: plt.Axes, r, z, v, v_min=-2, v_max=2):
    v = np.clip(v, v_min, v_max)
    cset_v = ax.contourf(r, z, v)
    cbi_v = figure.colorbar(cset_v, orientation="horizontal", shrink=0.8)
    cbi_v.set_label("Potential, V")


def plot_vector_field(figure: plt.Figure, ax: plt.Axes, r, z, er, ez, n=8):
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


def plot_one_layer(el: ElectrodeFields, parameters):
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

    plot_scalar_field(figure, ax, el.r, el.z, el.v)

    plot_vector_field(figure, ax, el.r, el.z, el.er, el.ez)


def plot_two_layer(el: ElectrodeFields, parameters):
    sigma_1 = parameters[0]
    sigma_2 = parameters[1]
    d_1 = parameters[2]
    I = parameters[3]

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

    plot_scalar_field(figure, ax, el.r, el.z, el.v)

    plot_vector_field(figure, ax, el.r, el.z, el.er, el.ez)

    brd = ax.plot(r[0], d_1 * np.ones(r[0].shape), color="black")


def plot_sum_of_two(el_1: ElectrodeFields, el_2: ElectrodeFields, parameters, r_shift):
    sigma_1 = parameters[0]
    sigma_2 = parameters[1]
    d_1 = parameters[2]
    I = parameters[3]

    el_sum = el_1.add(el_2, r_shift)

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

    plot_scalar_field(figure, ax, el_sum.r, el_sum.z, el_sum.v)
    plot_vector_field(figure, ax, el_sum.r, el_sum.z, el_sum.er, el_sum.ez)
    brd = ax.plot(el_sum.r[0], d_1 * np.ones(el_sum.r[0].shape), color="black")


def plot_sensitivity(
    el_i_1: ElectrodeFields,
    el_i_2: ElectrodeFields,
    el_v_3: ElectrodeFields,
    el_v_4: ElectrodeFields,
    parameters,
    r_shifts,
):
    sigma_1 = parameters[0]
    sigma_2 = parameters[1]
    d_1 = parameters[2]
    I = parameters[3]

    j1_injected = el_i_1.add(el_i_2, r_shifts[0])
    j2_measured = el_v_3.add(el_v_4, r_shifts[2] - r_shifts[1])

    figure = plt.figure()

    ax = figure.add_subplot()
    ax.invert_yaxis()
    ax.set_title(
        r"Sensitivity field for two-layer model for $\sigma_1$={} Sm/m, $\sigma_2$={} Sm/m, $d_1$={} m and a pair of electrodes with current I={} A and distance between them L = {} m and a pair of measurement electrodes with distance s = {} m".format(
            sigma_1, sigma_2, d_1, I, r_shifts[0], r_shifts[2] - r_shifts[1]
        )
    )

    plot_scalar_field(figure, ax, j1_injected.r, j1_injected.z, j1_injected.dot_product(j2_measured), -3, 10)
    brd = ax.plot(j1_injected.r[0], d_1 * np.ones(j1_injected.r[0].shape), color="black")


if __name__ == "__main__":
    model = "one-layer-model"
    v, er, ez, r, z, parameters = read_model_fields(model)
    el_0 = ElectrodeFields(v, er, ez, r, z)
    plot_one_layer(el_0, parameters)
    model = "two-layer-model"
    v, er, ez, r, z, parameters = read_model_fields(model)
    el_1 = ElectrodeFields(v, er, ez, r, z)
    el_2 = ElectrodeFields(-v, -er, -ez, r, z)
    plot_two_layer(el_1, parameters)
    plot_sum_of_two(el_1, el_2, parameters, 0.7)
    el_3 = ElectrodeFields(v, er, ez, r, z)
    el_4 = ElectrodeFields(-v, -er, -ez, r, z)
    plot_sensitivity(el_1, el_2, el_3, el_4, parameters, [0.1, 0.2, 0.3])

    plt.show()
