import numpy as np
import imageio

from skimage import color

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os


def difference_image(image1, image2, theta=200):
    diff = np.abs(image1 - image2)
    diff /= np.max(diff)

    diff = np.multiply(diff, 255)

    diff = diff.astype(int)
    threshold_indices = diff <= theta
    diff[threshold_indices] = 0

    return diff


def main():
    # 1. Bilder einlesen
    fig = plt.figure()
    # f, axarr = plt.subplots(1, 2)


    imgs = []
    diff_imgs = []

    for i in range(0, 5):
        filename = "cam00_%06i.jpg" % i
        rgb_img = imageio.imread(os.path.join("./04-tracking/input", filename))
        grayscale_img = (0.3 * rgb_img[:, :, 0]) + (0.59 * rgb_img[:, :, 1]) + (0.11 * rgb_img[:, :, 2])
        imgs.append(grayscale_img)

        diff_filename = "diff0_%06i.jpg" % (i+1)
        diff_img = imageio.imread(os.path.join("./04-tracking/intermediate", diff_filename))
        diff_imgs.append(diff_img)
    
    # 2. Über Folge iterieren
    movement_imgs = []
    for i in range(0, 4):

        # 2a. Differenzbild berechnen
        movement_img = difference_image(imgs[i], imgs[i+1])

        plt_im = plt.imshow(movement_img, animated=True)
        
        # 2b. Differenzbild speichern
        movement_imgs.append(plt_im)

        # f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(movement_img)
        # axarr[1].imshow(diff_imgs[i])

        # plt.imshow(movement_img - diff_imgs[i])

        # plt.show()


    # 3. Differenzbilder anzeigen
    ani = animation.ArtistAnimation(fig, movement_imgs)
    plt.show()


if __name__ == '__main__':
    main()
