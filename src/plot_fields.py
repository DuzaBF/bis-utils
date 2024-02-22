import numpy as np
import matplotlib.pyplot as plt
import scipy

EPS = 0.0001


class ElectrodeFields:

    def __init__(self, v, er, ez, r, z, r_shift=0, name="", v_name="Potential, V", e_name="Strength, V/m"):
        if v is not None:
            v = np.array(v, np.float64)
        if er is not None:
            er = np.array(er, np.float64)
            er = scipy.ndimage.median_filter(er, (3, 3))
        if ez is not None:
            ez = np.array(ez, np.float64)
            ez = scipy.ndimage.median_filter(ez, (3, 3))
        self.v = v
        self.er = er
        self.ez = ez
        self.r = r + r_shift
        self.z = z
        self.r_shift = r_shift
        self.name = name
        self.v_name = v_name
        self.e_name = e_name

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

    @staticmethod
    def cut_field(field, r_shift_index):
        if r_shift_index == 0:
            return field
        left_cut_index = int(r_shift_index / 2)
        right_cut_index = int(r_shift_index / 2) + int(r_shift_index % 2)

        if left_cut_index + right_cut_index != r_shift_index:
            raise ValueError("incorrect split")

        return field[:,   left_cut_index:-right_cut_index]

    def get_extended(self, r_shift_index):
        r_right = np.linspace(
            self.r_coords[-1] + self.r_step,
            self.r_coords[-1] + (r_shift_index + 1) * self.r_step,
            r_shift_index,
        )

        r_extended = np.concatenate([self.r_coords, r_right])
        r_ext, z_ext = np.meshgrid(r_extended, self.z_coords)

        return r_ext, z_ext

    def __add__(self, other):
        if type(other) != type(self):
            raise TypeError("Can only add two {}".format(type(self).__name__))
        sum_name = self.name + " + " + other.name
        if (self.r_step - other.r_step > EPS) or (self.z_step - other.z_step > EPS):
            raise ValueError("Fields must be on the same grid")

        if self.r_shift > other.r_shift:
            return other + self

        r_shift_index = int(
            np.ceil(abs(self.r_shift - other.r_shift) // self.r_step))

        r_ext, z_ext = self.get_extended(r_shift_index)

        v_sum = ElectrodeFields.sum_shifted_fields(
            self.v, other.v, r_shift_index)
        er_sum = ElectrodeFields.sum_shifted_fields(
            self.er, other.er, r_shift_index)
        ez_sum = ElectrodeFields.sum_shifted_fields(
            self.ez, other.ez, r_shift_index)

        r_cut = ElectrodeFields.cut_field(r_ext, r_shift_index)
        z_cut = ElectrodeFields.cut_field(z_ext, r_shift_index)
        v_cut = ElectrodeFields.cut_field(v_sum, r_shift_index)
        er_cut = ElectrodeFields.cut_field(er_sum, r_shift_index)
        ez_cut = ElectrodeFields.cut_field(ez_sum, r_shift_index)

        center = - (self.r_shift + other.r_shift) / 2

        return ElectrodeFields(v_cut,
                               er_cut,
                               ez_cut,
                               r_cut,
                               z_cut,
                               center,
                               sum_name)

    def dot_product(self, other):
        if type(other) != type(self):
            raise TypeError(
                "Can only multiply two {}".format(type(self).__name__))
        dot_name = self.name + " . " + other.name
        if (self.r_step - other.r_step > EPS) or (self.z_step - other.z_step > EPS):
            raise ValueError("Fields must be on the same grid")

        if self.r_shift > other.r_shift:
            return other.dot_product(self)

        r_shift_index = int(
            np.ceil(abs(self.r_shift - other.r_shift) // self.r_step))

        r_ext, z_ext = self.get_extended(r_shift_index)

        er_right = ElectrodeFields.extend_field_right(self.er, r_shift_index)
        er_left = ElectrodeFields.extend_field_left(other.er, r_shift_index)

        ez_right = ElectrodeFields.extend_field_right(self.ez, r_shift_index)
        ez_left = ElectrodeFields.extend_field_left(other.ez, r_shift_index)

        sensitivity = np.multiply(er_right, er_left) + \
            np.multiply(ez_right, ez_left)

        r_cut = ElectrodeFields.cut_field(r_ext, r_shift_index)
        z_cut = ElectrodeFields.cut_field(z_ext, r_shift_index)

        center = - (self.r_shift + other.r_shift) / 2

        return ElectrodeFields(sensitivity,
                               None,
                               None,
                               r_cut,
                               z_cut,
                               center,
                               dot_name)

    def norm(self):
        norm = np.sqrt(np.add(np.power(self.er, 2), np.power(self.ez, 2)))
        norm_name = "|"+self.name+"|"
        return ElectrodeFields(norm,
                               None,
                               None,
                               self.r - self.r_shift,
                               self.z,
                               self.r_shift,
                               norm_name)

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


def draw_scalar_field(figure: plt.Figure, ax: plt.Axes, r, z, v, v_min=np.NaN, v_max=np.NaN, label="Potential, V"):
    new_v = np.clip(v, v_min, v_max)
    cset_v = ax.contourf(r, z, new_v)
    cbi_v = figure.colorbar(cset_v, orientation="horizontal", shrink=0.8)
    cbi_v.set_label(label)
    return cset_v, cbi_v


def draw_vector_field(figure: plt.Figure, ax: plt.Axes, r, z, er, ez, n=8, label="Strength, V/m"):
    if (er is None) or (ez is None):
        return None
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
    return q_e, cbi_e


def draw_horizontal_line(figure: plt.Figure, ax: plt.Axes, r, d):
    brd = ax.plot(r[0], d * np.ones(r[0].shape), color="black")
    return brd


def plot_layer_model(el: ElectrodeFields, parameters: dict,  v_min=np.NaN, v_max=np.NaN, title="Electric field"):
    sigma = parameters.get("sigma") or 0
    sigma_1 = parameters.get("sigma_1")
    sigma_2 = parameters.get("sigma_2")
    d_1 = parameters.get("d_1") or 0
    I = parameters.get("I")

    figure = plt.figure()

    ax = figure.add_subplot()
    ax.invert_yaxis()
    ax.set_title(
        title
    )
    ax.set_xlabel("r, m")
    ax.set_ylabel("z, m")

    draw_scalar_field(figure, ax, el.r, el.z, el.v, v_min=v_min, v_max=v_max,
                      label=el.v_name)
    draw_vector_field(figure, ax, el.r, el.z, el.er, el.ez, label=el.e_name)
    draw_horizontal_line(figure, ax, el.r, d_1)
    draw_horizontal_line(figure, ax, el.r, 0)
    ax.set_xlim(el.r[0][0], el.r[0][-1])
    return figure


def plot_sensitivity(
    el_A: ElectrodeFields,
    el_B: ElectrodeFields,
    el_M: ElectrodeFields,
    el_N: ElectrodeFields,
    parameters: dict,
):
    sigma = parameters.get("sigma") or 0
    sigma_1 = parameters.get("sigma_1")
    sigma_2 = parameters.get("sigma_2")
    d_1 = parameters.get("d_1") or 0
    I = parameters.get("I")

    j1_injected = el_A + el_B
    j1_norm = j1_injected.norm()
    j1_norm.v_name = "Field strength, V/m"
    j2_measured = el_M + el_N
    j2_norm = j2_measured.norm()
    j2_norm.v_name = "Field strength, V/m"
    sensitivity = j1_injected.dot_product(j2_measured)
    sensitivity.v_name = "Sensitivity, m^-4"

    plot_layer_model(j1_injected, parameters, v_min=-2, v_max=2,
                     title="Field {}".format(j1_injected.name))
    plot_layer_model(j1_norm, parameters, v_min=-2, v_max=2,
                     title="Field {}".format(j1_norm.name))

    plot_layer_model(j2_measured, parameters, v_min=-2, v_max=2,
                     title="Field {}".format(j2_measured.name))
    plot_layer_model(j2_norm, parameters, v_min=-2, v_max=2,
                     title="Field {}".format(j2_norm.name))
    plot_layer_model(sensitivity, parameters, v_min=-2, v_max=2,
                     title="Sensitivity field")

if __name__ == "__main__":
    model = "two-layer-model"
    v, er, ez, r, z, parameters = read_model_fields(model)

    el_A = ElectrodeFields(v, er, ez, r, z, -0.2, "el_A")
    el_B = ElectrodeFields(-v, -er, -ez, r, z, 0.2, "el_B")
    el_M = ElectrodeFields(v, er, ez, r, z, -0.1, "el_M")
    el_N = ElectrodeFields(-v, -er, -ez, r, z, 0.1, "el_N")

    plot_sensitivity(el_A,
                     el_B,
                     el_M,
                     el_N, parameters)

    plt.show()
