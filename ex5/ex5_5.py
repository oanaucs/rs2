import functools

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from ex5_4 import GaussPDF


np.random.seed(0)


def load_points(filename):
    with open(filename) as file:
        point_strings = file.readlines()
    num_classes = int(point_strings[0].split(' ')[0])
    points_x = []
    points_y = []
    for i in range(1, len(point_strings)):
        pair = point_strings[i].split(' ')
        x = float(pair[0])
        y = float(pair[-1].split('\n')[0])
        points_x.append(x)
        points_y.append(y)

    # points_x /= np.max(points_x)
    # points_y /= np.max(points_y)

    points = list(zip(points_x, points_y))

    return num_classes, points


def assign_random_solution(num_classes, points):
    pdfs = []
    max_range = np.max(points)
    for i in range(0, num_classes):
        class_mean = np.random.random((1, 2))[0] * max_range
        c = np.random.random((1, 4))[0] * max_range * 1.5
        c = np.reshape(c, (2, 2))
        class_cov = np.transpose(c) @ c
        pdfs.append(GaussPDF(class_mean, class_cov))
    return pdfs


def iterate(num_classes, points, pdfs):
    probabilities = np.zeros((len(points), num_classes))
    for i in range(0, len(points)):
        for j in range(0, num_classes):
            # print('pdf', j)
            probabilities[i, j] = pdfs[j].density(points[i])
    # normalize probabilities such that row sum = 1
    for i in range(len(probabilities)):
        probabilities[i] = probabilities[i]/np.sum(probabilities[i])

    classes = np.argmax(probabilities, axis=1)
    classes = np.reshape(classes, (len(probabilities)))

    for i in range(0, num_classes):
        class_idxs = np.argwhere(classes==i)
        class_points = np.squeeze(np.take(points, class_idxs, axis=0))
        # print('newly assigned points', i, class_points.shape)
        new_class_mean = np.mean(class_points, axis=0)
        new_class_cov = np.cov(np.transpose(class_points))
        pdfs[i].update_mean(new_class_mean)
        pdfs[i].update_cov(new_class_cov)


def main():
    filename = './05-em/input/2dist_complex.txt'
    num_classes, points = load_points(filename)
    pdfs = assign_random_solution(num_classes, points)
    for i in range(0, 100):
        probabilities = iterate(num_classes, points, pdfs)
    for i in range(0, num_classes):
        print('pdf', i)
        print('mean', pdfs[i].mean)
        print('cov', pdfs[i].cov)


if __name__ == '__main__':
    main()
