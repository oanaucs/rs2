from rs2_util import to_homogenous
import numpy as np
import imageio
import re

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rs2_util import vec2d, to_homogenous, from_homogenous

def load_F(F_file):
    F = np.ones((3, 3))

    with open(F_file) as file:
        pars=file.readlines()
        row = 0
        for p in pars:
            par=p.split(' ')
            par[-1]=par[-1].rstrip('\n')
            for col in range(0, 3):
                F[row, col] = par[col]
            row += 1
    return F

def compute_epipoles(F):
    u, d, v = np.linalg.svd(F, full_matrices=False)
    right_epipole = v[-1, :]
    right_epipole /= right_epipole[-1]

    u, d, v = np.linalg.svd(np.transpose(F), full_matrices=False)
    left_epipole = v[-1, :]
    left_epipole/= left_epipole[-1]

    return (left_epipole, right_epipole)


def main():
    np.set_printoptions(3, suppress=True, linewidth=160)

    # 1. Bild laden
    img = imageio.imread('./05-reconstruct/input/seq000100.ppm')
    f_file = './05-reconstruct/input/fund_000_100.txt'

    img_x = img.shape[1]
    img_y = img.shape[0]

    F = load_F(f_file)
    compute_epipole(F)


if __name__ == "__main__":
    main()
