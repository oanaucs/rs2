import numpy as np
import imageio
import matplotlib.pyplot as plt

from rs2_util import vec2d, to_homogenous, from_homogenous
from ex1_4 import calculate_H


def main():
    # 1. Bild laden und anzeigen
    ...

    # 2. Punkte x1, ..., x4 anklicken lassen
    image_points = ...
    world_points = ..
    ...

    # 3. Projektive Abbildung berechen
    H = calculate_H(image_points, world_points)

    # 4. Neues Bild anlegen
    ...

    # 5. Transformation für jeden Punkt des neuen Bildes ausführen (vgl. Aufgabe 2)
    for newy, newx in np.ndindex(newimg.shape[:2]):
        ...

    # 6. Neues Bild anzeigen
    ...


if __name__ == '__main__':
    main()
