import math
import numpy as np
import imageio
import skimage.transform
import matplotlib.pyplot as plt

from rs2_util import vec2d
from ex1_2 import rotate_img_around_point


def main():
    # 1. Bild laden und anzeigen
    fig = plt.figure()
    img = mpimg.imread('stinkbug.png')

    img_shape = img.shape()

    # 2. Mittelpunkt p berechnen und auf Bild anzeigen
    p = (img_shape[0] / 2, img_shape[1] / 2)
    plt.plot([p[0]], [p[1]], marker='x')

    # 3. Punkt q anklicken lassen
    q = []

    def onclick(event):
        if len(q) < 1:
            ix, iy = event.xdata, event.ydata
            q.append(vec2d(ix, iy))
            plt.plot(ix, iy, ',')
            fig.canvas.draw()


    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    
    plt.imshow(img)
    plt.show()
    q = q[0]
    alpha = math.atan2(q[1] - p[1], q[0] - p[0])

    # 4. Transformation mit Funktion aus Ausgabe 2 ausführen
    newimg = rotate_img_around_point(img, p, q)

    # 5. Transformation mit skimage-Funktion ausführen
    sk_newimg = skimage.transform.rotate(img, alpha)

    # 6. Differenzbild anzeigen
    diff_img = sk_newimg - newimg

    # 7. MSE berechnen und ausgeben
    print('MSE: %f' % ...)



if __name__ == '__main__':
    main()
