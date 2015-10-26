import numpy as np
import cv2
import glob
import os
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def main():
    # copy parameters to arrays
    K = np.array([[1743.23312, 0, 2071.06177], [0, 1741.57626, 1476.48298], [0, 0, 1]])
    d = np.array([-0.307412748, 0.300929683, 0, 0, 0])  # just use first two terms (no translation)

    os.mkdir("photos/undistort")

    logging.debug("Starting loop")
    for image in glob.glob("photos/*.JPG"):
        logging.debug("var %s", image)
        imgname = image.split("\\")[1]
        logging.debug("Undistorting %s . . . ", imgname)
        # read one of your images
        img = cv2.imread(image)
        h, w = img.shape[:2]

        # un-distort
        newcamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 0)
        newimg = cv2.undistort(img, K, d, None, newcamera)

        # cv2.imwrite("original.jpg", img)
        cv2.imwrite("photos/undistort/" + imgname, newimg)

if __name__ == '__main__':
    main()
