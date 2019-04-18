import math
import numpy as np
import imageio
import skimage.transform
import matplotlib.pyplot as plt

from rs2_util import vec2d
from ex1_2 import rotate_img_around_point


def main():
    # 1. Bild laden und anzeigen
    ...

    # 2. Mittelpunkt p berechnen und auf Bild anzeigen
    p = ...
    plt.plot([p[0]], [p[1]], marker='x')

    # 3. Punkt q anklicken lassen
    ...
    q = ...
    alpha = ...

    # 4. Transformation mit Funktion aus Ausgabe 2 ausführen
    ...

    # 5. Transformation mit skimage-Funktion ausführen
    ...

    # 6. Differenzbild anzeigen
    ...

    # 7. MSE berechnen und ausgeben
    print('MSE: %f' % ...)



if __name__ == '__main__':
    main()
