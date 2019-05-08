import math
import imageio

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import skimage.feature


def main():
    # 1. Bild laden
    fig = plt.figure()
    img = imageio.imread('../material/02-orb/input/hotel/cmu-long-hotel-010.pgm')
    # 2. Features extrahieren
    feature_extractor = skimage.feature.ORB(n_keypoints=200)

    feature_extractor.detect_and_extract(img)
    keypoints = feature_extractor.keypoints
    scales = feature_extractor.scales
    orientations = feature_extractor.orientations
    
    # normalize scales
    scales = scales / max(scales) * 50
    # 3. Bild anzeigen
    plt.imshow(img)
    # 4. Features visualisieren

    #TODO fix
    for i in range(0, len(keypoints)):
        arrow_x = scales[i] * math.cos(orientations[i])
        arrow_y = scales[i] * math.sin(orientations[i])
        plt.arrow(keypoints[i][1], keypoints[i][0], arrow_x, arrow_y)


    plt.show()


if __name__ == '__main__':
    main()
