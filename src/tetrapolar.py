import two_layer_model
import one_layer_model
import plot_fields
from typing import Union
import numpy as np


class Tetrapolar:
    def __init__(self,
                 model: Union[one_layer_model.OneLayerModel, two_layer_model.TwoLayerModel],
                 el_coords: list,
                 ) -> None:
        self.model = model
        self.el_coords = el_coords
        pass

    def impedance(self, current_pair: tuple, measure_pair: tuple) -> np.float64:
        el_A = self.el_coords[current_pair[0]-1]
        el_B = self.el_coords[current_pair[1]-1]
        el_M = self.el_coords[measure_pair[0]-1]
        el_N = self.el_coords[measure_pair[1]-1]

        v_M = self.model.field_potential_surface(
            1, el_M - el_A) + self.model.field_potential_surface(-1, el_M - el_B)
        v_N = self.model.field_potential_surface(
            1, el_N - el_A) + self.model.field_potential_surface(-1, el_N - el_B)

        u_MN = v_N - v_M

        imp = abs(u_MN) / 1

        return imp

    def plot(self, current_pair: tuple, measure_pair: tuple):
        pass


if __name__ == "__main__":
    t = Tetrapolar(two_layer_model.TwoLayerModel(1, 2, 0.1), [0, 0.1, 0.2, 0.3])
    print("Z_23 ", t.impedance((1, 4), (2, 3)))
    print("Z_12 ", t.impedance((3, 4), (1, 2)))
    print("Z_13 ", t.impedance((2, 4), (1, 3)))
