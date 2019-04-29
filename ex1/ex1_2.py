import math
import numpy as np
import imageio

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('./../')
from rs2_util import vec2d


def f_inv(p, q, s):
    """Inverse Transformation zu ex1_1.f (vgl. Aufgabe 1)."""

    alpha = math.atan2(q[1] - p[1], q[0] - p[0])
    R_inv = np.matrix([[math.cos(alpha), math.sin(alpha)], [-math.sin(alpha), math.cos(alpha)]])
    f_s_inv = np.matmul(R_inv, s - p) + p
    f_s_inv = f_s_inv.reshape((2, 1))
    return (np.asscalar(f_s_inv[0]), np.asscalar(f_s_inv[1]))


def rotate_img_around_point(img, p, q):
    # 1. Neues Bild anlegen
    newimg = np.zeros_like(img)

    # 2. Transformation für jeden Punkt des neuen Bildes ausführen
    for newy, newx in np.ndindex(newimg.shape[:2]):
        oldx, oldy = f_inv(p, q, (newx, newy))
        oldx = int(oldx)
        oldy = int(oldy)
        if (0 < oldy and oldy < newimg.shape[0] and 0 < oldx and oldx < newimg.shape[1]):
            newimg[newy, newx] = img[oldy, oldx]

    # 3. Neues Bild zurückgeben
    return newimg


def main():
    # 1. Bild laden und anzeigen
    fig = plt.figure()
    img = mpimg.imread('01_rotation/input/stinkbug.png')

    # 2. Punkte p und q anklicken lassen
    plt.imshow(img)

    clicks = plt.ginput(2)

    plt.show()

    p, q = np.asarray(clicks[0]), np.asarray(clicks[1])
    print(p)

    # 3. Transformation ausführen
    newimg = rotate_img_around_point(img, p, q)

    # 5. Neues Bild anzeigen
    plt.imshow(newimg)
    plt.show()

if __name__ == '__main__':
    main()
