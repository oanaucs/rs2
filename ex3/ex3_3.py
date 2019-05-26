import math
import numpy as np
import imageio
import re

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from rs2_util import to_homogenous
from ex3_2 import calculate_F, calculate_F_hat

N = 1000
d_thresh = 0.0001


def d_squared(p, l):
    l_hat = (1 / np.sqrt(np.square(l[0]) + np.square(l[1]))) * l
    p_hat = (1 / p[2]) * p
    return np.square(np.dot(l_hat,np.transpose(p_hat)))


def RANSAC_F(img_l_points, img_r_points):
    # 1. Stichproben bestimmen
    samples = []
    for i in range(0, N):
        idxs = np.random.randint(0, len(img_l_points), 8)
        samples_l = [img_l_points[i] for i in idxs]
        samples_r = [img_r_points[i] for i in idxs]
        samples.append((samples_l, samples_r))

    # 2. Alle Stichproben testen
    max_num = 0
    F_star = None
    for i in range(0, N):
        # 2a. Fundamentalmatrix für Stichprobe bestimmen
        F_hat = calculate_F_hat(calculate_F(samples[i][0], samples[i][1]))
        if F_star is None:
            F_star = F_hat

        # 2b. Distanzen bestimmen um Anzahl unterstützender Punkte auszurechnen
        num_p = 0
        for j in range(0, len(samples[i][0])):
            p_l = to_homogenous(samples[i][0][j])
            p_r = to_homogenous(samples[i][1][j])
            l_l = F_hat @ p_l
            l_r = F_hat @ p_r

            e = d_squared(p_l, l_r) + d_squared(p_r, l_l)
            if (e < d_thresh):
                num_p += 1
        if (num_p >= max_num):
            max_num = num_p
            F_star = F_hat

    print('supporting points number', max_num)
        
    return F_star

def normalize(x, min_x, max_x):
    return (x-min_x)/(max_x-min_x) * 2 - 1

def main():
    np.set_printoptions(3, suppress=True, linewidth=160)

    # 1. Bilder laden
    img0 = imageio.imread('./03-epipolar/input/img2_0.jpg')
    img1 = imageio.imread('./03-epipolar/input/img2_1.jpg')

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img0)
    axarr[1].imshow(img1)

    img_x = img0.shape[1]
    img_y = img0.shape[0]

    # 2. Punkte laden
    with open('./03-epipolar/input/img2_points.txt') as file:
        point_strings=file.readlines()
    img_l_points=[]
    img_r_points=[]
    for p in point_strings:
            pairs=p.split(' ')
            x0 = np.float(pairs[0])
            y0 = np.float(pairs[1])

            x1 = np.float(pairs[-2])
            y1 = np.float(pairs[-1])

            img_l_points.append((normalize(x0, 0, img_x), normalize(y0, 0, img_y)))
            img_r_points.append((normalize(x1, 0, img_x), normalize(y1, 0, img_y)))

            axarr[0].plot(x0, y0, 'bo')
            axarr[1].plot(x1, y1, 'bo')

    # plt.show()

    # 3. RANSAC-Algorithmus
    F = RANSAC_F(img_l_points, img_r_points)

    print('F:\n%s' % str(F))

    for i in range(0, 5):
        print((to_homogenous(img_r_points[i]) @ F @ to_homogenous(img_l_points[i])))


if __name__ == "__main__":
    main()
