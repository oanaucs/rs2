import sys
sys.path.append('./../')

from rs2_util import vec2d, to_homogenous, from_homogenous
import numpy as np
import imageio

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def A_rows_for_single_pair(x, x_):
    A = np.zeros((2, 9))
    A[0] = [0, 0, 0, -x_[2] * x[0], -x_[2] * x[1], -x_[2]
            * x[2], x_[1] * x[0], x_[1] * x[1], x_[1] * x[2]]
    A[1] = [x_[2] * x[0], x_[2] * x[1], x_[2] * x[2], 0,
            0, 0, -x_[0] * x[0], -x_[0] * x[1], -x_[0] * x[2]]
    return A


def calculate_H(image_points, world_points):
    # 1. Koordinaten in homogene Koordinaten transformieren
    image_points = [to_homogenous(i) for i in image_points]
    world_points = [to_homogenous(i) for i in world_points]

    A_rows = [A_rows_for_single_pair(x_, x) for x_, x in zip(image_points, world_points)]
    A = np.vstack(A_rows)

    # 3. Homogenes Gleichungssystem Ah=0 lösen
    u, d, v = np.linalg.svd(A)

    h = v[-1, :]
    h /= h[-1]

    # 4. Projektive Abbildung berechnen und zurückgeben
    H = np.reshape(h, (3, 3))
    return H


def main():
    # 1. Bild laden und anzeigen
    fig = plt.figure()
    img = mpimg.imread('01_projective/input/stinkbug.png')

    # 2. Punkte x1, ..., x4 anklicken lassen
    plt.imshow(img)

    clicks = plt.ginput(4)

    plt.show()

    image_points = [vec2d(i[0], i[1]) for i in clicks]
    world_points = [vec2d(0, 0), vec2d(0, 1), vec2d(1, 0), vec2d(1, 1)]

    # 3. Projektive Abbildung berechnen
    H = calculate_H(image_points, world_points)

    print('Projektive Abbildung:')
    print(H)

    # 4. Sanity Check durch Anwenden der Projektion
    world_points = [to_homogenous(i) for i in world_points]
    image_points = [H *i for i in world_points]


if __name__ == '__main__':
    main()
