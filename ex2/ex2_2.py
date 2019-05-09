import numpy as np
import imageio

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches
import skimage.feature


def d(desc1, desc2):
    """ Ermittelt den Abstand zwischen zwei Deskriptoren."""
    # compute hamming distance for binary descriptors
    if len(desc1) != len(desc2):
        return None
    
    distance = 0
    for i in range(0, len(desc1)):
        if desc1[i] != desc2[i]:
            distance += 1   
    return distance


def find_correspondences(descs1, descs2):
    """ Ermittelt Korrespondenzen zwischen desc1 und desc2 und gibt diese als Liste von Indexpaaren zurück."""
    corr = []
    for i in range(0, len(descs1)):
        distances = [d(descs1[i], j) for j in descs2]
        indexes = np.argsort(distances)[:2]
        ratio = distances[indexes[0]] / distances[indexes[1]]
        if ratio > 0.8:
            corr.append((i, indexes[0]))
    return corr


def main():
     # 1. Bild laden
    img1 = imageio.imread('../material/02-orb/input/hotel/cmu-long-hotel-010.pgm')
    img2 = imageio.imread('../material/02-orb/input/hotel/cmu-long-hotel-015.pgm')

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img1)
    axarr[1].imshow(img2)


    # 2. Features extrahieren
    feat1 = skimage.feature.ORB(n_keypoints=50)
    feat2 = skimage.feature.ORB(n_keypoints=50)

    feat1.detect_and_extract(img1)
    feat2.detect_and_extract(img2)

    keypoints1 = feat1.keypoints
    keypoints2 = feat2.keypoints

    # 3. Korrespondenzen finden
    corr = find_correspondences(feat1.descriptors, feat2.descriptors)

    print('found correspondences:', len(corr))

    #TODO fix missing line ends
    # 4. Korrespondenzen visualisieren
    for i in range(0, len(corr)):
        k1 = keypoints1[corr[i][0]]
        k2 = keypoints2[corr[i][1]]

        p1 = [k1[1], k1[0]]
        p2 = [k2[1], k2[0]]

        axarr[0].plot(p1[0], p1[1], 'bo')
        axarr[1].plot(p2[0], p2[1], 'bo')
        
        con = matplotlib.patches.ConnectionPatch(xyA=p2, xyB=p1, coordsA='data', coordsB='data',
                      axesA=axarr[1], axesB=axarr[0], color='#53F242',
                      arrowstyle="<-", shrinkB=3)
        axarr[1].add_artist(con)


    plt.show()


if __name__ == '__main__':
    main()
