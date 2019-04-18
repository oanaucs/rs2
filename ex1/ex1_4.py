import numpy as np
import imageio
import matplotlib.pyplot as plt

from rs2_util import vec2d, to_homogenous, from_homogenous


def A_rows_for_single_pair(x, x_):
    A = np.zeros((2, 9))
    A[0] = [0, 0, 0, -x_[2] * x[0], -x_[2] * x[1], -x_[2] * x[2], x_[1] * x[0], x_[1] * x[1], x_[1] * x[2]]
    A[1] = [x_[2] * x[0], x_[2] * x[1], x_[2] * x[2], 0, 0, 0, -x_[0] * x[0], -x_[0] * x[1], -x_[0] * x[2]]
    return A


def calculate_H(image_points, world_points):
    # 1. Koordinaten in homogene Koordinaten transformieren
    ...

    # 2. A-Matrix ausrechnen
    ...

    # 3. Homogenes Gleichungssystem Ah=0 lösen
    ...

    # 4. Projektive Abbildung berechnen und zurückgeben
    H = ...
    return H


def main():
    # 1. Bild laden und anzeigen
    ...

    # 2. Punkte x1, ..., x4 anklicken lassen
    image_points = ...
    world_points = ...

    # 3. Projektive Abbildung berechnen
    H = calculate_H(image_points, world_points)

    print('Projektive Abbildung:')
    print(H)

    # 4. Sanity Check durch Anwenden der Projektion



if __name__ == '__main__':
    main()
