import numpy as np
import imageio

from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import functools

from ex5_1 import load_F
from ex5_2 import compute_camera_matrices
from rs2_util import vec2d, to_homogenous, from_homogenous

import os

def normalize(x, min_x, max_x):
    return (x-min_x)/(max_x-min_x) * 2 - 1

def A_rows_for_point(x, P):
    A0 = [x[1]*P[2,0]-x[2]*P[1,0], x[1]*P[2,1]-x[2]*P[1,1], x[1]*P[2,2]-x[2]*P[1,2], x[1]*P[2,3]-x[2]*P[1,3]]

    A1 = [x[2]*P[0,0]-x[0]*P[2,0], x[2]*P[0,1]-x[0]*P[2,1], x[2]*P[0,2]-x[0]*P[2,2], x[2]*P[0,3]-x[0]*P[2,3]]

    return np.vstack((A0, A1))

def project_back(X_x, X_y, X_z, P):
    projected_points_x = []
    projected_points_y = []

    for i in range(0, len(X_x)):
        X = to_homogenous([X_x[i], X_y[i], X_z[i]])
        x = from_homogenous(P @ X)
        print([X_x[i], X_y[i], X_z[i]], x)
        projected_points_x.append(x[0])
        projected_points_y.append(x[1])

    return projected_points_x, projected_points_y


def main():
    # 1. Punkte laden
    img = imageio.imread('./05-reconstruct/input/seq000050.ppm')
    plt.imshow(img)

    img_x = img.shape[1]
    img_y = img.shape[0]

    print('x', img_x, 'y', img_y)

    with open('./05-reconstruct/input/corr_000_050.unnorm.txt') as file:
        point_strings = file.readlines()
    img_l_points = []
    img_r_points = []
    for p in point_strings:
        pairs = p.split(' ')

        x0 = np.float(pairs[0])
        y0 = np.float(pairs[1])

        x1 = np.float(pairs[-2])
        y1 = np.float(pairs[-1])

        img_l_points.append((x0, y0))
        img_r_points.append((x1, y1))

    # 2. Kamera Matritzen erzeugen
    f_file = './05-reconstruct/input/fund_000_050.txt'
    F = load_F(f_file)
    
    (P0, P1) = compute_camera_matrices(F)

    P2 = P1 + (np.ones(P1.shape))
   
    # 3. Triangularisierung
    proj_x_0 = []
    proj_y_0 = []

    proj_x_1 = []
    proj_y_1 = []

    proj_x_2 = []
    proj_y_2 = []

    for i in range(0, len(img_l_points)):
        A0 = A_rows_for_point(to_homogenous(img_l_points[i]), P0)
        A1 = A_rows_for_point(to_homogenous(img_r_points[i]), P1)

        A = np.vstack((A0, A1))

        u, d, v = np.linalg.svd(A)

        X = v[-1, :]

        x0 = from_homogenous(P0 @ X)

        proj_x_0.append(x0[0])
        proj_y_0.append(x0[1])

        x1 = from_homogenous(P1 @ X)

        proj_x_1.append(x1[0])
        proj_y_1.append(x1[1])

        x2 = from_homogenous(P2 @ X)

        proj_x_2.append(x2[0])
        proj_y_2.append(x2[1])


    plt.plot(proj_x_0, proj_y_0, 'bo')
    plt.plot(proj_x_1, proj_y_1, 'bo', color='gold')
    plt.plot(proj_x_2, proj_y_2, 'bo', color='red')


    plt.show()



if __name__ == '__main__':
    main()
