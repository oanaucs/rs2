import numpy as np
import imageio

from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import functools

from ex4_1 import difference_image
from ex4_2 import center_of_gravity
from rs2_util import vec2d, to_homogenous, from_homogenous

import os

def normalize(x, min_x, max_x):
    return (x-min_x)/(max_x-min_x) * 2 - 1

def main():
    # 1. Kalibrierungsdaten lesen
    img0 = imageio.imread('./04-tracking/input/cam00_000000.jpg')
    img1 = imageio.imread('./04-tracking/input/cam01_000000.jpg')

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img0)
    axarr[1].imshow(img1)


    img_x = img0.shape[1]
    img_y = img0.shape[0]

    print(img_x, img_y)

    o0 = (int(img_x/2), int(img_y/2)) 

    o1 = (int(img_x/2), int(img_y/2)) 

    calib0 = './04-tracking/input/calib0_norm.txt'
    calib1 = './04-tracking/input/calib1_norm.txt'

    P0 = load_calibration_data(calib0, o0)
    P1 = load_calibration_data(calib1, o1)

    #             # plot corresponding points
    #         axarr[0].plot(y0, x0, 'bo')
    #         axarr[1].plot(y1, x1, 'bo')

    # plt.show()

    # 2. Bilder einlesen
    filenames0 = []
    filenames1 = []
    imgs_0 = []
    imgs_1 = []
    for i in range(10):
        filename0 = "cam00_%06i.jpg" % i
        filename1 = "cam01_%06i.jpg" % i

        filenames0.append(os.path.join("./04-tracking/input", filename0))
        filenames1.append(os.path.join("./04-tracking/input", filename1))

    # 3. Differenzbilder und Schwerpunkte berechnen
    plots, centroids0, centroids1 = calculate_and_plot(None, (filenames0, filenames1), o0)

    # 4. Triangularisierung
    points = []
    for i in range(0, 1):
        # A0 = A_rows_for_point(to_homogenous(centroids0[i]), P0)
        # A1 = A_rows_for_point(to_homogenous(centroids1[i]), P1)

        # A = np.vstack((A0, A1))

        # u, d, v = np.linalg.svd(A)

        # X = v[-1, :]
        # X /= X[-1]

        # X = from_homogenous(X)

        # print(X)
        # print(from_homogenous(P0 @ to_homogenous(X)), centroids0[i])

        X = [10.285564, 6.952230, 16.008560]

        print(from_homogenous(P0 @ to_homogenous(X)), [627.622969, 28.960521])


    # # 5. Differenzbilder anzeigen
    # ...

    # # 6. 3D-Trajektorie anzeigen
    # ...


def A_rows_for_point(x, P):
    A0 = [x[1]*P[2,0]-x[2]*P[1,0], x[1]*P[2,1]-x[2]*P[1,1], x[1]*P[2,2]-x[2]*P[1,2], x[1]*P[2,3]-x[2]*P[1,3]]

    A1 = [x[2]*P[0,0]-x[0]*P[2,0], x[2]*P[0,1]-x[0]*P[2,1], x[2]*P[0,2]-x[0]*P[2,2], x[2]*P[0,3]-x[0]*P[2,3]]

    return np.vstack((A0, A1))


def load_calibration_data(calib_file, o):
    R = np.ones((3, 3))
    t = np.ones((3, 1))
    f = 0.0

    with open(calib_file) as file:
        pars=file.readlines()
        for p in pars:
            par=p.split(' ')
            par[-1]=par[-1].rstrip('\n')
            if 'r' in par[0]:
                pos = [int(s) - 1 for s in list(par[0]) if s.isdigit()]
                R[pos[0], pos[1]] = float(par[1])
            if 't' in par[0]:
                if par[0] == 'tx':
                    t[0, 0] = float(par[1])
                if par[0] == 'ty':
                    t[1, 0] = float(par[1])
                if par[0] == 'tz':
                    t[2, 0] = float(par[1])
            else:
                f = float(par[1])
    
    R_t = np.concatenate((R, t), axis = 1)
    # print(R_t)

    K = np.zeros((3, 3))
    K[0, 0] = f
    K[0, 2] = o[0]
    K[1, 1] = f
    K[1, 2] = o[1]
    K[2, 2] = 1

    return np.matmul(K, R_t)


def calculate_and_plot(ax, image_filenames, default_value_for_cog=None):
    imgs_0 = []
    imgs_1 = []
    for i in range(10):
        rgb_img0 = imageio.imread(image_filenames[0][i])
        grayscale_img0 = (0.3 * rgb_img0[:, :, 0]) + (0.59 * rgb_img0[:, :, 1]) + (0.11 * rgb_img0[:, :, 2])
        imgs_0.append(grayscale_img0)
        rgb_img1 = imageio.imread(image_filenames[1][i])
        grayscale_img1 = (0.3 * rgb_img1[:, :, 0]) + (0.59 * rgb_img1[:, :, 1]) + (0.11 * rgb_img1[:, :, 2])
        imgs_1.append(grayscale_img1)


    diff_imgs0 = []
    diff_imgs1 = []
    centroids0 = []
    centroids1 = []
    # 1. Über Folge iterieren
    for i in range(0, 2):
        # 1a. Differenzbild berechnen
        diff_img0 = difference_image(imgs_0[i], imgs_0[i+1], theta=25)
        diff_imgs0.append(diff_img0)

        diff_img1 = difference_image(imgs_1[i], imgs_1[i+1], theta=160)
        diff_imgs1.append(diff_img1)

        # 1b. Schwerpunkt berechnen
        centroid0 = center_of_gravity(diff_img0)
        centroids0.append(centroid0)

        centroid1 = center_of_gravity(diff_img1)
        centroids1.append(centroid1)

        # 1c. Schön darstellen
        plots = None

    return plots, centroids0, centroids1


if __name__ == '__main__':
    main()
