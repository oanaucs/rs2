import numpy as np
import imageio

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ex4_1 import difference_image

import os

def calculate_moment(diff_image, i, j):
    moment = 0
    for y in range(0, diff_image.shape[0]):
        for x in range(0, diff_image.shape[1]):
            moment += np.power(x, i) * np.power(y, j) * diff_image[y, x]
    return moment


def center_of_gravity(diff_image, default_value=None):
    m00 = calculate_moment(diff_image, 0, 0)
    m10 = calculate_moment(diff_image, 1, 0)
    if (np.isclose(m00, 0.0)):
        return default_value
    else:
        center_x = m10 / m00
        m01 = calculate_moment(diff_image, 0, 1)
        center_y = m01 / m00
        return (center_x, center_y)
    return default_value

def main():
    # 1. Bilder einlesen
    fig = plt.figure()

    imgs = []
    for i in range(0, 5):
        filename = "cam01_%06i.jpg" % i
        rgb_img = imageio.imread(os.path.join("./04-tracking/input", filename))
        grayscale_img = (0.3 * rgb_img[:, :, 0]) + (0.59 * rgb_img[:, :, 1]) + (0.11 * rgb_img[:, :, 2])
        imgs.append(grayscale_img)

    # 2. Über Folge iterieren
    movement_imgs = []
    centroids = []
    for i in range(0, 5):
        # 2a. Differenzbild berechnen
        movement_img = difference_image(imgs[i], imgs[i+1], theta=150)
        plt_im = plt.imshow(movement_img, animated=True)

        # 2b. Schwerpunkt berechnen
        center_x, center_y = center_of_gravity(movement_img, (int(movement_img.shape[1]/2), int(movement_img.shape[0]/2)))
        centroids.append((center_x, center_y))
        

        print(center_x, center_y)
        plt.plot(center_x, center_y, 'bo')
        plt.show()
    # 3. Differenzbilder anzeigen
    ...


if __name__ == '__main__':
    main()
