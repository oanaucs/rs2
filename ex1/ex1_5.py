import numpy as np
import imageio
import matplotlib.pyplot as plt

from rs2_util import vec2d, to_homogenous, from_homogenous
from ex1_4 import calculate_H


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

    # 3. Projektive Abbildung berechen
    H = calculate_H(image_points, world_points)

    # 4. Neues Bild anlegen
    undistimg = np.zeros(img.shape)

    # 5. Transformation für jeden Punkt des neuen Bildes ausführen (vgl. Aufgabe 2)
    for undisty, undistx in np.ndindex(undistimg.shape[:2]):
        (oldx, oldy) = np.inverse(H) * (undistx, undisty)
        undistimg[undistx, undisty] = img[oldx, oldy]

    # 6. Neues Bild anzeigen
    ...


if __name__ == '__main__':
    main()
