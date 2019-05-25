import numpy as np
import imageio
import re

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from rs2_util import vec2d, to_homogenous, from_homogenous
from ex3_3 import RANSAC_F

def within_bounds(x):
    return (x[0] >= -1 and x[0] <= 1 and x[1] >= -1 and x[1] <= 1)

def compute_points_on_line(l, img_x, img_y):
    left = (-1, (-l[2] + l[0]) / l[1])
    right = (1, (-l[2] - l[0]) / l[1])

    print('left right', within_bounds(left), within_bounds(right))

    if (within_bounds(left) and within_bounds(right)):
        left = (unnormalize(left[0], 0, img_x), unnormalize(left[1], 0, img_y))
        right = (unnormalize(right[0], 0, img_x), unnormalize(right[1], 0, img_y))

        return ((left[0], right[0]), (left[1], right[1]))

    bottom = ((-l[2] + l[1]) / l[0], -1)
    top = ((-l[2] - l[1]) / l[0], 1)
    print('bottom top', within_bounds(bottom), within_bounds(top))

    if (within_bounds(bottom) and within_bounds(top)):
        bottom = (unnormalize(bottom[0], 0, img_x), unnormalize(bottom[1], 0, img_y))
        top = (unnormalize(top[0], 0, img_x), unnormalize(top[1], 0, img_y))

        return ((bottom[0], top[0]), (bottom[1], top[1]))

    return None

def normalize(x, min_x, max_x):
    return (x-min_x)/(max_x-min_x) * 2 - 1

def unnormalize(x, min_x, max_x):
    return (x+1)/2 * (max_x - min_x) + min_x

def main():
    np.set_printoptions(3, suppress=True, linewidth=160)

    # 1. Bilder laden
    img0 = imageio.imread('./03-epipolar/input/img1_0.jpg')
    img1 = imageio.imread('./03-epipolar/input/img1_1.jpg')

    img_x = img0.shape[1]
    img_y = img0.shape[0]

    # 2. Punkte laden
    with open('./03-epipolar/input/img1_points.txt') as file:
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

    # 3. RANSAC-Algorithmus
    F = RANSAC_F(img_l_points, img_r_points)
    print("F:\n%s" % str(F))

    # 4. Bilder anzeigen
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img0)
    axarr[1].imshow(img1)

    num_points = 2
    curr_num = 0

    while True:
	# 4a. Punkte anklicken lassen
        pts = plt.ginput(num_points)

        for p in pts:
            x_x = normalize(p[0], 0, img_x)
            x_y = normalize(p[1], 0, img_y)

            x = (x_x, x_y)

            # 4b. Epipolarlinie berechnen
            l = F @ to_homogenous(x)

            points = compute_points_on_line(l, img_x, img_y)

            print(points)

            axarr[1].plot(points[0], points[1])

        plt.show()


if __name__ == "__main__":
    main()
