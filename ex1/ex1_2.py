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
    f_s_inv = R_inv * np.expand_dims(np.transpose(np.asarray(s - p)), axis=1) + np.expand_dims(np.transpose(np.asarray(p)), axis=1)
    return np.asscalar(f_s_inv[0]), np.asscalar(f_s_inv[1])


def rotate_img_around_point(img, p, q):
    # 1. Neues Bild anlegen
    newimg = np.zeros_like(img)

    # 2. Transformation für jeden Punkt des neuen Bildes ausführen
    for newy, newx in np.ndindex(newimg.shape[:2]):
        oldy, oldx = f_inv(p, q, (newy, newx))
        oldx = math.trunc(oldx)
        oldy = math.trunc(oldy)
        if (oldy < newimg.shape[0] and oldx < newimg.shape[1]):
            if (newy < newimg.shape[0] and newx < newimg.shape[1]):
                newimg[newy, newx] = img[oldy, oldx]

    # 3. Neues Bild zurückgeben
    return newimg


def main():
    # 1. Bild laden und anzeigen
    fig = plt.figure()
    img = mpimg.imread('stinkbug.png')

    # 2. Punkte p und q anklicken lassen
    clicks = []

    def onclick(event):
        if len(clicks) < 2:
            ix, iy = event.xdata, event.ydata
            clicks.append(vec2d(ix, iy))
            plt.plot(ix, iy, ',')
            fig.canvas.draw()


    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    
    plt.imshow(img)
    plt.show()
    # TODO
    # if len(clicks) == 2:
    #     fig.canvas.mpl_disconnect(cid)

    p, q = clicks[0], clicks[1]

    # 3. Transformation ausführen
    newimg = rotate_img_around_point(img, p, q)
    print(newimg)

    # 5. Neues Bild anzeigen
    plt.imshow(newimg)
    plt.show()

if __name__ == '__main__':
    main()
