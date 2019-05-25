import numpy as np
import imageio
import re

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rs2_util import to_homogenous


def A_row_for_single_pair(x, x_):
    return [x_[0] * x[0], x_[0]*x[1], x_[0], x_[1]*x[0], x_[1]*x[1], x_[1], x[0], x[1], 1]


def calculate_F(pls, prs):
    A = np.zeros((len(pls), 9))
    for i in range(len(pls)):
        A[i] = A_row_for_single_pair(pls[i], prs[i])

    u, d, v = np.linalg.svd(A, full_matrices=False)

    f = v[-1, :]
    f /= f[-1]

    return np.reshape(f, (3,3))

def calculate_F_hat(F):
    u, d, v = np.linalg.svd(F, full_matrices=False)

    d[-1] = 0
    F_hat = np.dot(u * d, v)
    return F_hat

def normalize(x, min_x, max_x):
    return (x-min_x)/(max_x-min_x) * 2 - 1

def main():
    np.set_printoptions(3, suppress=True, linewidth=160)

    # 1. Bilder laden
    img0 = imageio.imread('./03-epipolar/input/img1_0.jpg')
    img1 = imageio.imread('./03-epipolar/input/img1_1.jpg')

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img0)
    axarr[1].imshow(img1)

    img_x = img0.shape[1]
    img_y = img0.shape[0]

    # 2. Punkte laden
    with open('./03-epipolar/input/img1_points.txt') as file:
        point_strings=file.readlines()
    img0_points=[]
    img1_points=[]
    for p in point_strings:
            pairs=p.split(' ')
            x0 = np.float(pairs[0])
            y0 = np.float(pairs[1])

            x1 = np.float(pairs[-2])
            y1 = np.float(pairs[-1])

            img0_points.append((normalize(x0, 0, img_x), normalize(y0, 0, img_y)))
            img1_points.append((normalize(x1, 0, img_x), normalize(y1, 0, img_y)))

            axarr[0].plot(x0, y0, 'bo')
            axarr[1].plot(x1, y1, 'bo')

    # plt.show()

    # 3. F mit Rang 3 bestimmen
    F = calculate_F(img0_points[:8], img1_points[:8])
    # check rang
    rang_F = np.linalg.matrix_rank(F)
    print('rang F', rang_F)

    # 4. Rang von F reduzieren
    F_hat = calculate_F_hat(F)
    rang_F_hat = np.linalg.matrix_rank(F_hat)
    print('rang F hat', rang_F_hat)


if __name__ == "__main__":
    main()
