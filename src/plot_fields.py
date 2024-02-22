import numpy as np
import matplotlib.pyplot as plt


class ElectrodeFields:

    def __init__(self, v, er, ez, r, z, r_shift=0):
        v = np.array(v, np.float64)
        er = np.array(er, np.float64)
        ez = np.array(ez, np.float64)
        self.v = v
        self.er = er
        self.ez = ez
        self.r = r
        self.z = z
        self.r_shift = r_shift

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
        right = np.empty((len(field), r_shift_index))
        right.fill(np.nan)
        return np.concatenate([field, right], 1)

    @staticmethod
    def extend_field_left(field, r_shift_index):
        left = np.empty((len(field), r_shift_index))
        left.fill(np.nan)
        return np.concatenate([left, field], 1)

    @staticmethod
    def sum_shifted_fields(field_1, field_2, r_shift_index):
        return ElectrodeFields.extend_field_right(
            field_1, r_shift_index
        ) + ElectrodeFields.extend_field_left(field_2, r_shift_index)

    def __add__(self, other):
        if type(other) != type(self):
            raise TypeError("Can only add two {}".format(type(self).__name__))
        if (self.r_step != other.r_step) or (self.z_step != other.z_step):
            raise ValueError("Fields must be on the same grid")

        if self.r_shift > other.r_shift:
            return other + self

        r_shift_index = int(
            np.ceil(abs(self.r_shift - other.r_shift) // self.r_step))

        r_right = np.linspace(
            self.r_coords[-1] + self.r_step,
            self.r_coords[-1] + (r_shift_index + 1) * self.r_step,
            r_shift_index,
        )

        print(self.r_shift)
        print(other.r_shift)
        print(r_shift_index)

        r_extended = np.concatenate([self.r_coords, r_right])
        r_ext, z_ext = np.meshgrid(r_extended, self.z_coords)

        v_sum = ElectrodeFields.sum_shifted_fields(
            self.v, other.v, r_shift_index)
        er_sum = ElectrodeFields.sum_shifted_fields(
            self.er, other.er, r_shift_index)
        ez_sum = ElectrodeFields.sum_shifted_fields(
            self.ez, other.ez, r_shift_index)

        r_cut = r_ext[:,   :-r_shift_index]
        z_cut = z_ext[:,   :-r_shift_index]
        v_cut = v_sum[:,   :-r_shift_index]
        er_cut = er_sum[:, :-r_shift_index]
        ez_cut = ez_sum[:, :-r_shift_index]

        return ElectrodeFields(v_cut, er_cut, ez_cut, r_cut, z_cut, -(self.r_shift + other.r_shift) / 2)

    def dot_product(self, other):
        if type(other) != type(self):
            raise TypeError(
                "Can only multiply two {}".format(type(self).__name__))
        if (self.r_step != other.r_step) or (self.z_step != other.z_step):
            raise ValueError("Fields must be on the same grid")

        if self.r_shift > other.r_shift:
            return other.dot_product(self)

        r_shift_index = int(
            np.ceil(abs(self.r_shift - other.r_shift) // self.r_step))

        er_right = ElectrodeFields.extend_field_right(self.er, r_shift_index)
        er_left = ElectrodeFields.extend_field_left(other.er, r_shift_index)

        ez_right = ElectrodeFields.extend_field_right(self.ez, r_shift_index)
        ez_left = ElectrodeFields.extend_field_left(other.ez, r_shift_index)

        return er_right * er_left + ez_right * ez_left

    def invert(self):
        self.v = -self.v
        self.er = -self.er
        self.ez = -self.ez
        self.r = -self.r
        self.z = -self.z


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
    parameters = np.load(
        "./computed/{}/parameters.npy".format(model), allow_pickle=True
    ).item()
    r = np.genfromtxt("./computed/{}/r.csv".format(model),
                      delimiter=",", usemask=True)
    z = np.genfromtxt("./computed/{}/z.csv".format(model),
                      delimiter=",", usemask=True)

    return v, er, ez, r, z, parameters


def plot_scalar_field(figure: plt.Figure, ax: plt.Axes, r, z, v, v_min=-2, v_max=2, label="Potential, V"):
    v = np.clip(v, v_min, v_max)
    cset_v = ax.contourf(r, z, v)
    cbi_v = figure.colorbar(cset_v, orientation="horizontal", shrink=0.8)
    cbi_v.set_label(label)


def plot_vector_field(figure: plt.Figure, ax: plt.Axes, r, z, er, ez, n=4, label="Strength, V/m"):
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
    cbi_e.set_label(label)


def plot_one_layer(el: ElectrodeFields, parameters: dict):
    sigma = parameters.get("sigma")
    I = parameters.get("I")

    figure = plt.figure()

    ax = figure.add_subplot()
    ax.invert_yaxis()
    ax.set_title(
        r"Electric field for one-layer model for I={} A, $\sigma$={} Sm/m".format(
            I, sigma
        )
    )
    ax.set_xlabel("r, m")
    ax.set_ylabel("z, m")

    plot_scalar_field(figure, ax, el.r, el.z, el.v)
    plot_vector_field(figure, ax, el.r, el.z, el.er, el.ez)
    ax.set_xlim(el.r[0][0], el.r[0][-1])


def plot_two_layer(el: ElectrodeFields, parameters: dict):
    sigma_1 = parameters.get("sigma_1")
    sigma_2 = parameters.get("sigma_2")
    d_1 = parameters.get("d_1")
    I = parameters.get("I")

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

    norm = np.sqrt(np.add(np.power(el.er, 2), np.power(el.ez, 2)))
    plot_scalar_field(figure, ax, el.r + el.r_shift, el.z, norm)

    # plot_scalar_field(figure, ax, el.r, el.z, el.v)
    # plot_vector_field(figure, ax, el.r, el.z, el.er, el.ez)
    brd = ax.plot(r[0], d_1 * np.ones(r[0].shape), color="black")
    ax.set_xlim(el.r[0][0], el.r[0][-1])


def plot_sensitivity(
    el_A: ElectrodeFields,
    el_B: ElectrodeFields,
    el_M: ElectrodeFields,
    el_N: ElectrodeFields,
    parameters: dict,
):
    sigma_1 = parameters.get("sigma_1")
    sigma_2 = parameters.get("sigma_2")
    d_1 = parameters.get("d_1")
    I = parameters.get("I")

    j1_injected = el_A + el_B
    j2_measured = el_M + el_N

    plot_two_layer(j1_injected, parameters)
    plot_two_layer(j2_measured, parameters)

    sensitivity = j1_injected.dot_product(j2_measured)

    figure = plt.figure()

    ax = figure.add_subplot()
    ax.invert_yaxis()
    ax.set_title(
        r"Sensitivity field for two-layer model for $\sigma_1$={} Sm/m, $\sigma_2$={} Sm/m, $d_1$={} m and a pair of electrodes with current I={} A and distance between them L = {} m and a pair of measurement electrodes with distance s = {} m".format(
            sigma_1, sigma_2, d_1, I, el_B.r_shift -
            el_A.r_shift, el_M.r_shift - el_N.r_shift
        )
    )

    plot_scalar_field(figure, ax, el_A.r, el_A.z,
                      sensitivity, -10, 10, "Sensitivity")
    brd = ax.plot(el_A.r[0], d_1 *
                  np.ones(el_A.r[0].shape), color="black")
    ax.set_xlim(el_A.r[0][0], el_A.r[0][-1])


class TetrapolarSystem:

    def __init__(self, v, er, ez, r, z, I, x: list):
        '''
           A   M  N    B
        ---|---|--|----|---

        -> x
        '''
        self.el_A = ElectrodeFields(v, er, ez, r, z, x[0])
        self.el_M = ElectrodeFields(v, er, ez, r, z, x[1])
        self.el_N = ElectrodeFields(v, er, ez, r, z, x[2])
        self.el_B = ElectrodeFields(v, er, ez, r, z, x[3])


if __name__ == "__main__":
    model = "two-layer-model"
    v, er, ez, r, z, parameters = read_model_fields(model)

    el_A = ElectrodeFields(v, er, ez, r, z, -0.2)
    el_B = ElectrodeFields(-v, -er, -ez, r, z, 0.2)
    el_M = ElectrodeFields(v, er, ez, r, z, -0.1)
    el_N = ElectrodeFields(-v, -er, -ez, r, z, 0.1)

    plot_sensitivity(el_A,
                     el_B,
                     el_M,
                     el_N, parameters)

    plt.show()
