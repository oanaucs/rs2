from rs2_util import to_homogenous
import numpy as np
import imageio
import re

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ex5_1 import load_F, compute_epipoles

def compute_P0():
    return np.concatenate((np.identity(3), np.zeros((3, 1))), axis=1)

def compute_skew_symmetric_matrix_from_vec(vec):
    mat = np.zeros((3, 3))

    mat[0] = [0, -vec[2], vec[1]]
    mat[1] = [vec[2], 0, -vec[0]]
    mat[2] = [-vec[1], vec[0], 0]

    return mat

def compute_P1(right_epipole, F):
    # print('right epipole', right_epipole)
    e_right_mat = compute_skew_symmetric_matrix_from_vec(right_epipole)

    P1 = e_right_mat @ F
    right_epipole = np.expand_dims(right_epipole, axis=1)
    P1 = np.concatenate((P1, right_epipole), axis=1)
    # print('p1', P1)

    return P1

def compute_camera_matrices(F):
    epipoles = compute_epipoles(F)

    # debug
    # print(F @ epipoles[0], np.transpose(F) @ epipoles[1])

    P0 = compute_P0()
    # print('p0', P0)
    P1 = compute_P1(epipoles[1], F)

    return (P0, P1)


def main():
    np.set_printoptions(3, suppress=True, linewidth=160)

    # 1. Bild laden
    img = imageio.imread('./05-reconstruct/input/seq000100.ppm')

    img_x = img.shape[1]
    img_y = img.shape[0]

    # 2. Punkte laden
    f_file = './05-reconstruct/input/fund_000_100.txt'
    F = load_F(f_file)
    
    cam_matrices = compute_camera_matrices(F)

if __name__ == "__main__":
    main()
