import functools
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('TkAgg')


class GaussPDF():
    def __init__(self, mean, cov):
        self.mean = np.asarray(mean)
        self.cov = np.asarray(cov)
        self.d = self.cov.shape[0]

    def update_mean(self, mean):
        self.mean = mean

    def update_cov(self, cov):
        self.cov = cov

    def density(self, x):
        det = np.linalg.det(self.cov)
        term1 = 1 / np.sqrt(np.power(2*np.pi, self.d) * det)
        term2 = np.exp(-0.5 * np.transpose(x - self.mean) @ np.linalg.inv(self.cov) @ (x - self.mean))

        p_x = term1 *  term2
        return p_x

    def plot_2d_gauss(self, ax):
        u, d, v = np.linalg.svd(cov, full_matrices=False)

        angle = np.degrees(np.arctan(v[1] / v[0]))[1]

        lengths = [3 * np.sqrt(eigenval) for eigenval in d]

        ellipse = matplotlib.patches.Ellipse(mean,
                                             width=lengths[0],
                                             height=lengths[1],
                                             angle=angle)
        ellipse.set_facecolor('darkblue')

        return ax.add_artist(ellipse)


def main():
    # mean = np.asarray((1, 2))
    # cov = np.asarray([[1, 0], [0, 1]])

    mean = [2, 4]
    cov = [[0.5, 0.8], [0.8, 0.5]]

    plotter = GaussPlotter(mean, cov)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plotter.plot_2d_gauss(ax, mean, cov)
    plt.show()


if __name__ == '__main__':
    main()
