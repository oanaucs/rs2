import math
import numpy as np
import imageio
import matplotlib.pyplot as plt

from rs2_util import vec2d


def f_inv(p, q, s):
    """Inverse Transformation zu ex1_1.f (vgl. Aufgabe 1)."""

    alpha = ...
    R_inv = ...

    return ...


def rotate_img_around_point(img, p, q):
    # 1. Neues Bild anlegen
    newimg = np.zeros_like(img)

    # 2. Transformation für jeden Punkt des neuen Bildes ausführen
    for newy, newx in np.ndindex(newimg.shape[:2]):
        ...

    # 3. Neues Bild zurückgeben
    return newimg


def main():
    # 1. Bild laden und anzeigen
    ...

    # 2. Punkte p und q anklicken lassen
    ...
    p, q = v...

    # 3. Transformation ausführen
    newimg = rotate_img_around_point(img, p, q)

    # 5. Neues Bild anzeigen
    ...


if __name__ == '__main__':
    main()
