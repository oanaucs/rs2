from rs2_util import vec3d
import pytest
import math
import numpy as np
import imageio

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def A_row_for_single_pair(xs, ys, xw, yw):
    return np.asarray([xs * xw, xs * yw, xs, -ys * xw, -ys * yw, -ys])


def calculate_v(image_points, world_points):
    # 1. A-Matrix ausrechnen
    A = np.zeros((len(image_points), 6))
    for i in range(len(image_points)):
        A[i] = A_row_for_single_pair(
            image_points[i][0], image_points[i][1], world_points[i][0], world_points[i][1])

    # 2. Homogenes Gleichungssystem Av=0 lösen
    u, d, v = np.linalg.svd(A, full_matrices=False)

    s = v[-1, :]
    s /= s[-1]

    return s


def calculate_Rtxy(image_points, world_points):
    # 1. Ersten Teil der Parameter berechnen
    (r21_, r22_, ty_, r11_, r12_, tx_) = calculate_v(image_points, world_points)

    # 2. Gleichung 7 einsetzen
    b = -(r11_ * r11_ + r12_ * r12_ + r21_ * r21_ + r22_ * r22_)

    c = np.square(r11_ * r22_ - r21_ * r12_)

    k_squared = (-b + np.sqrt(b * b - 4 * c)) / 2

    k=np.sqrt(k_squared)

    # 3. Gleichung 8 einsetzen
    r13_squared=k_squared - r11_*r11_ - r12_*r12_
    r13_=np.sqrt(r13_squared)

    r23_squared=k_squared - r21_*r21_ - r22_*r22_
    r23_=np.sqrt(r23_squared)

    sgn23_= np.sign((r11_ * r21_ + r12_ * r22_) / r13_)
    if (r23_ * sgn23_ > 0):
        r23_= -r23_

    # 4. Vektoren normieren

    r1_=np.asarray([r11_, r12_, r13_]) / k
    r2_=np.asarray([r21_, r22_, r23_]) / k

    # 5. Vektor r3 bestimmen
    r3_=np.cross(r1_, r2_)

    # 6. Translationsparameter normieren
    tx_ = tx_ / k
    ty_ = ty_ / k

    # 7. Ausgabe
    R_=np.vstack((r1_, r2_, r3_))
    return R_, tx_, ty_


def main():
    np.set_printoptions(3, suppress=True, linewidth=160)

    # 1. Bild laden und anzeigen
    img=imageio.imread('../material/02-calib/input0/input.ppm')

    plt.imshow(img)

    # 2. Punkte laden und anzeigen
    with open('../material/02-calib/input0/points.txt') as f:
        point_strings=f.readlines()
    image_points=[]
    world_points=[]
    for p in point_strings:
        pairs=p.split(' ')
        pairs[-1]=pairs[-1].rstrip('\n')
        pairs=[np.float(i) for i in pairs]
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

    # 4. Parameter schätzen
    R_, tx_, ty_=calculate_Rtxy(new_image_points, world_points)

    print('R: ')
    print(R_)

    print('t: ')
    print((tx_, ty_))


if __name__ == '__main__':
    main()
