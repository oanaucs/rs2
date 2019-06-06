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

def normalize(x, min_x, max_x):
    return (x-min_x)/(max_x-min_x) * 2 - 1

def main():
    # 1. Kalibrierungsdaten lesen
    img0 = imageio.imread('./04-tracking/input/cam00_000000.jpg')
    img1 = imageio.imread('./04-tracking/input/cam01_000000.jpg')

    img_x = img0.shape[1]
    img_y = img0.shape[0]

    o0 = (int(img_x/2), int(img_y/2)) 

    o1 = (int(img_x/2), int(img_y/2)) 

    calib0 = './04-tracking/input/calib0.txt'
    calib1 = './04-tracking/input/calib1.txt'

    P0 = load_calibration_data(calib0, o0)
    P1 = load_calibration_data(calib1, o1)

    img0_points = []
    img1_points = []

    with open('./04-tracking/intermediate/points.txt') as file:
        point_strings=file.readlines()
        image_points=[]
        world_points=[]
        for p in point_strings:
            pairs = p.split(' ')
            x0 = np.float(pairs[0])
            y0 = np.float(pairs[1])

            x1 = np.float(pairs[-2])
            y1 = np.float(pairs[-1])

            img0_points.append((x0, y0))
            img1_points.append((x1, y1))

            # img0_points.append((normalize(x0, 0, img_x), normalize(y0, 0, img_y)))
            # img1_points.append((normalize(x1, 0, img_x), normalize(y1, 0, img_y)))

    # # 2. Bilder einlesen
    # ...

    # # 3. Differenzbilder und Schwerpunkte berechnen
    # ...

    # # 4. Triangularisierung
    for i in range(0, 1):
        A0 = A_rows_for_point(to_homogenous(img0_points[i]), P0)
        A1 = A_rows_for_point(to_homogenous(img1_points[i]), P1)

        A = np.vstack((A0, A1))

        print(A.shape)

        u, d, v = np.linalg.svd(A, full_matrices=False)

        X = v[-1, :]
        X /= X[-1]

        print(from_homogenous(X))
    # # 5. Differenzbilder anzeigen
    # ...

    # # 6. 3D-Trajektorie anzeigen
    # ...


def A_rows_for_point(x, P):
    A0 = [x[1]*P[2,0]-x[2]*P[1,0], x[1]*P[2,1]-P[1,1], x[1]*P[2,2]-x[2]*P[1,2], x[1]*P[2,3]-x[2]*P[1,3]]

    A1 = [x[2]*P[0,0]-x[0]*P[2,0], x[2]*P[0,1]-x[0]*P[2,1], x[2]*P[0,2]-x[0]*P[2,2], x[2]*P[0,3]-x[0]*P[2,3]]
    return np.vstack((A0, A1))


def load_calibration_data(calib_file, o):
    R = np.ones((3, 4))
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
                    R[0, 3] = float(par[1])
                if par[0] == 'ty':
                    R[1, 3] = float(par[1])
                if par[0] == 'tx':
                    R[2, 3] = float(par[1])
            else:
                f = float(par[1])


    K = np.zeros((3, 3))
    K[0,0] = f
    K[0, 2] = o[0]
    K[1, 1] = f
    K[1, 2] = o[1]
    K[2, 2] = 1

    return np.matmul(K, R)

# def calculate_and_plot(ax, image_filenames, default_value_for_cog=None):
#     # 1. Über Folge iterieren
#     for ...

#         # 1a. Differenzbild berechnen
#         ...

#         # 1b. Schwerpunkt berechnen
#         ...

#         # 1c. Schön darstellen
#         ...

#     return plots, centers_of_gravity


if __name__ == '__main__':
    main()
