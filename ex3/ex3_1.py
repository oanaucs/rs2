import numpy as np
import imageio

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rs2_util import vec2d, to_homogenous, from_homogenous


def main():
    # 1. Bild laden und anzeigen
    fig = plt.figure()
    img = imageio.imread('./03-calib/input0/input.ppm')

    plt.imshow(img)

    img_center_x = int(img.shape[1] / 2)
    img_center_y = int(img.shape[0] / 2)

    # 2. Kameraparameter
    R = np.asarray([
        [-0.226322, 0.972695, 0.051411],
        [0.683302, 0.196159, -0.703292],
        [-0.694174, -0.124042, -0.709039],
    ])
    t = [7.115603, 0.824146, 46.850660]
    f = 1.684674  * 640

    # 3. Punkte laden
    with open('./03-calib/input0/points.txt') as file:
        point_strings=file.readlines()
    image_points=[]
    world_points=[]
    for p in point_strings:
        pairs=p.split(' ')
        pairs[-1]=pairs[-1].rstrip('\n')
        pairs=[np.float(i) for i in pairs]
        world_points.append((pairs[0], pairs[1]))
        image_points.append((pairs[2], pairs[3]))

        plt.plot(image_points[-1][0], image_points[-1][1], 'bo')

    # 4. Punkte projizieren
    computed_image_points = []
    for i in range(0, len(world_points)):
        # print(world_points[i])
        wc = np.asarray([world_points[i][0], world_points[i][1], 0])
        xc = np.dot(R[0], wc) + t[0]
        yc = np.dot(R[1], wc) + t[1]
        zc = np.dot(R[2], wc) + t[2]

        xs = f * (xc / zc)
        ys = f * (yc / zc)

        computed_image_points.append((xs + img_center_x, ys + img_center_y))

        print(xs + img_center_x, ys + img_center_y, image_points[i])

    # 5. Punkte anzeigen
    for p in computed_image_points:
        plt.plot(p[0], p[1], 'bo', color='red')

    plt.show()

    


if __name__ == "__main__":
    main()
