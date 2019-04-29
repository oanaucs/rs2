import numpy as np
import imageio

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('./../')
from rs2_util import vec2d, to_homogenous, from_homogenous
from ex1_4 import calculate_H


def main():
    # 1. Bild laden und anzeigen
    fig = plt.figure()
    img = mpimg.imread('01_projective/input/stinkbug.png')

    # 2. Punkte x1, ..., x4 anklicken lassen
    plt.imshow(img)

    clicks = plt.ginput(4)

    plt.show()

    image_points = [vec2d(i[0], i[1]) for i in clicks]
    world_points = [vec2d(0, 0), vec2d(1, 0), vec2d(0, 1), vec2d(1, 1)]

    # 3. Projektive Abbildung berechen
    H = calculate_H(image_points, world_points)

    # 4. Neues Bild anlegen
    newimg = np.zeros(img.shape)

    # 5. Transformation für jeden Punkt des neuen Bildes ausführen (vgl. Aufgabe 2)
    for newy, newx in np.ndindex(newimg.shape[:2]):
        homo_wc = to_homogenous(vec2d(newx, newy) / vec2d(newimg.shape[0], newimg.shape[1]))
        homo_ic = np.matmul(H, homo_wc)
        oldx, oldy = from_homogenous(homo_ic)
        oldx = int(oldx)
        oldy = int(oldy)
        if (oldx > 0 and oldx < newimg.shape[1] and oldy > 0 and oldy < newimg.shape[0]):
            newimg[newy, newx] = img[int(oldy), int(oldx)]

    # 6. Neues Bild anzeigen
    plt.imshow(newimg)
    plt.show()


if __name__ == '__main__':
    main()
