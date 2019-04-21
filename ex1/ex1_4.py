import numpy as np
import imageio
import matplotlib.pyplot as plt

from rs2_util import vec2d, to_homogenous, from_homogenous


def A_rows_for_single_pair(x, x_):
    A = np.zeros((2, 9))
    A[0] = [0, 0, 0, -x_[2] * x[0], -x_[2] * x[1], -x_[2] * x[2], x_[1] * x[0], x_[1] * x[1], x_[1] * x[2]]
    A[1] = [x_[2] * x[0], x_[2] * x[1], x_[2] * x[2], 0, 0, 0, -x_[0] * x[0], -x_[0] * x[1], -x_[0] * x[2]]
    return A


def calculate_H(image_points, world_points):
    # 1. Koordinaten in homogene Koordinaten transformieren
    image_points = [to_homogenous(i) for i in image_points]
    world_points = [to_homogenous(i) for i in world_points]

    # 2. A-Matrix ausrechnen
    A = np.zeros((2*len(image_points), 9))
    for i in range(len(image_points)):
        sub_A = A_rows_for_single_pair(world_points[i], image_points[i])
        A[2 * i] = sub_A[0]
        A[2 * i + 1] = sub_A[1] 

    # 3. Homogenes Gleichungssystem Ah=0 lösen
    u, d, v = np.linalg.svd(A)

    # 4. Projektive Abbildung berechnen und zurückgeben
    H = np.reshape(v, (3, 3))
    return H


def main():
    # 1. Bild laden und anzeigen
    fig = plt.figure()
    img = mpimg.imread('stinkbug.png')

    # 2. Punkte x1, ..., x4 anklicken lassen
    clicks = []

    def onclick(event):
        if len(clicks) < 4:
            ix, iy = event.xdata, event.ydata
            clicks.append(vec2d(ix, iy))
            plt.plot(ix, iy, ',')
            fig.canvas.draw()


    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    
    plt.imshow(img)
    plt.show()

    image_points = [vec2d(i[0], i[1]) for i in clicks]
    world_points = [vec2d(0, 0), vec2d(0, 1), vec2d(1, 0), vec2d(1, 1)]

    # 3. Projektive Abbildung berechnen
    H = calculate_H(image_points, world_points)

    print('Projektive Abbildung:')
    print(H)

    # 4. Sanity Check durch Anwenden der Projektion
    image_points = H * world_points


if __name__ == '__main__':
    main()
