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
    x = np.linspace(-1,1,100)
    y = [(-l[2]+l[0]*x_t)/l[1] for x_t in x]

    x = np.array([unnormalize(x_t, 0, img_x) for x_t in x])
    y = np.array([unnormalize(y_t, 0, img_y) for y_t in y])
    idxs_in_img = (y>=0) & (y<img_y) 

    return (x[idxs_in_img], y[idxs_in_img])

def normalize(x, min_x, max_x):
    return (x-min_x)/(max_x-min_x) * 2 - 1

def unnormalize(x, min_x, max_x):
    return (x+1)/2 * (max_x - min_x) + min_x

def main():
    np.set_printoptions(3, suppress=True, linewidth=160)

    # 1. Bilder laden
    img0 = imageio.imread('./03-epipolar/input/img2_0.jpg')
    img1 = imageio.imread('./03-epipolar/input/img2_1.jpg')

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

    # 3. RANSAC-Algorithmus
    F = RANSAC_F(img_l_points, img_r_points)
    print("F:\n%s" % str(F))

    # 4. Bilder anzeigen
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img0)
    axarr[1].imshow(img1)

    while True:
	# 4a. Punkte anklicken lassen
        pts = plt.ginput(1)

        for p in pts:
            x_x = normalize(p[0], 0, img_x)
            x_y = normalize(p[1], 0, img_y)

            x = (x_x, x_y)

            # 4b. Epipolarlinie berechnen
            l = F @ to_homogenous(x)

            points = compute_points_on_line(l, img_x, img_y)

            if points is not None:
                axarr[1].plot(points[0], points[1])


if __name__ == "__main__":
    main()
