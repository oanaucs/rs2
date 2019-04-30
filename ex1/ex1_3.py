import math
import numpy as np
import imageio
import skimage.transform

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from rs2_util import vec2d
from ex1_2 import rotate_img_around_point


def main():
    # 1. Bild laden und anzeigen
    img = mpimg.imread('01_rotation/input/stinkbug.png')
    img_shape = img.shape

    # 2. Mittelpunkt p berechnen und auf Bild anzeigen
    p = np.asarray([int(img_shape[1] / 2), int(img_shape[0] / 2)])
    marker_style = dict(color='red', linestyle=':', marker='x',
                    markersize=15, markerfacecoloralt='gray')
    plt.plot([p[0]], [p[1]], **marker_style)

    # 3. Punkt q anklicken lassen
    plt.imshow(img)

    q = plt.ginput(1)

    plt.show()

    q = q[0]

    alpha = math.atan2(q[1] - p[1], q[0] - p[0])

    # 4. Transformation mit Funktion aus Ausgabe 2 ausführen
    newimg = np.asarray(rotate_img_around_point(img, p, q), dtype=np.float32)
    plt.imshow(newimg)
    plt.show()

    # 5. Transformation mit skimage-Funktion ausführen
    sk_newimg = skimage.transform.rotate(img, -np.degrees(alpha))

    plt.imshow(sk_newimg)
    plt.show()    

    # 6. Differenzbild anzeigen
    diff_img = sk_newimg.astype(np.float32) - newimg

    plt.imshow(diff_img)
    plt.show()

    # 7. MSE berechnen und ausgeben
    mse = 0
    for y, x in np.ndindex(newimg.shape[:2]):
        mse += np.square(sk_newimg[y, x] - newimg[y, x])
    mse = 1 / (newimg.shape[0] * newimg.shape[1]) * mse

    mse = np.mean(np.square(newimg - sk_newimg))
    print('MSE: %f' % mse)



if __name__ == '__main__':
    main()
