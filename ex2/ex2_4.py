import math
import numpy as np
import imageio
import pytest

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rs2_util import vec2d, vec3d

from ex2_3 import calculate_Rtxy


def A_row_for_single_pair(xs, ys, xw, yw, R_, tx_, ty_):
    return np.asarray([ys, -(R_[1,0] * xw + R_[1,1]*yw + ty_)])


def b_row_for_single_pair(xs, ys, xw, yw, R_, tx_, ty_):
    return np.asarray([- ys*R_[2,0]*xw - ys*R_[2,1]*yw])


def calculate_full_Rtf(image_points, world_points, R_, tx_, ty_):
    # 1. Gleichungssystem (10) aufstellen
    A = np.zeros((len(image_points), 2))
    for i in range(len(image_points)):
        A[i] = A_row_for_single_pair(
            image_points[i][0], image_points[i][1], world_points[i][0], world_points[i][1], R_, tx_, ty_)


    b = np.zeros((len(image_points), 1))
    for i in range(len(image_points)):
        b[i] = b_row_for_single_pair(
            image_points[i][0], image_points[i][1], world_points[i][0], world_points[i][1], R_, tx_, ty_)

    # 2. Gleichungssystem lösen
    u, s, v = np.linalg.svd(A)

    s_inv = np.zeros((A.shape[0], A.shape[1])).T
    s_inv[:s.shape[0], :s.shape[0]] = np.linalg.inv(np.diag(s))

    A_pinv = v.T.dot(s_inv).dot(u.T)
    
    x = A_pinv @ b

    f_ = np.asscalar(x[1])

    # 3. Vektor t zusammenfügen

    t_ = (tx_, ty_, np.asscalar(x[0]))

    # 4. Ggf. Vorzeichen anpassen
    if (f_ < 0):
        # anpassen des Vorzeichens r23_
        R_[1,2] = -R_[1, 2]
        R_[0, 2] = -R_[0, 2]
        R_[2, 0] = -R_[2, 0]
        R_[2, 1] = -R_[2, 1]

        # sanity check 
        print('signs match', np.sign(R_[1, 2] * R_[0, 2]) == - np.sign((r11_ * r21_ + r12_ * r22_)))

    # 5. Ausgabe
    return R_, t_, f_


def main():
    np.set_printoptions(3, suppress=True, linewidth=160)

    # 1. Bild laden und anzeigen
    img=imageio.imread('../material/02-calib/input1/input.ppm')

    plt.imshow(img)

    # 2. Punkte laden und anzeigen
    with open('../material/02-calib/input1/points.txt') as f:
        point_strings=f.readlines()
    image_points=[]
    world_points=[]
    for p in point_strings:
        pairs=p.split(' ')
        pairs[-1]=pairs[-1].rstrip('\n')
        pairs=[float(i) for i in pairs]
        world_points.append((pairs[0], pairs[1]))
        image_points.append((pairs[2], pairs[3]))

        plt.plot(image_points[-1][0], image_points[-1][1], 'bo')

    plt.show()

    # 3. Bild- und Weltkoordinaten ermitteln (gegeben sind Pixel- und Weltkoordinaten)
    in_max_x = img.shape[1]
    in_max_y = img.shape[0]

    new_image_points=[]
    for p in image_points:
        xs = p[0] - int(in_max_x / 2)
        ys = p[1] - int(in_max_y / 2)
        new_image_points.append((np.around(xs, 3), np.around(ys, 3)))

    # 4. Parameter R und tx, ty schätzen
    R_, tx_, ty_ = calculate_Rtxy(new_image_points, world_points)
    R_, t_, f_ = calculate_full_Rtf(new_image_points, world_points, R_, tx_, ty_)

    print('R: ')
    print(R_)

    print('t: ')
    print(t_)

    print('f: ')
    print(f_)


if __name__ == '__main__':
    main()
