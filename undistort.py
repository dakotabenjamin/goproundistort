import numpy as np
import cv2
import os
import logging
import pyexiv2

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

#TODO: calculate and copy new ccd_width, fix thumbnail
def main(file_list, outdir):
    # copy parameters to arrays
    K = np.array([[1743.23312, 0, 2071.06177], [0, 1741.57626, 1476.48298], [0, 0, 1]])
    d = np.array([-0.307412748, 0.300929683, 0, 0, 0])  # just use first two terms (no translation)

    logging.debug("Starting loop")
    for image in file_list:
        logging.debug("var %s", image)
        imgname = image.split("/")[-1]
        new_image_path = os.path.join(outdir, imgname)
        if not os.path.exists(new_image_path):
            logging.debug("Undistorting %s . . . ", imgname)
            # read one of your images
            img = cv2.imread(image)
            h, w = img.shape[:2]

            # un-distort
            newcamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 0)
            newimg = cv2.undistort(img, K, d, None, newcamera)

            # cv2.imwrite("original.jpg", img)
            cv2.imwrite(new_image_path, newimg)

            # Write metadata
            old_meta = pyexiv2.ImageMetadata(image)
            new_meta = pyexiv2.ImageMetadata(new_image_path)
            old_meta.read()
            new_meta.read()
            old_meta.copy(new_meta)
            new_meta.write()
        else:
            logging.debug("Image already processed")


def get_args():
    import argparse
    p = argparse.ArgumentParser(description='Remove Distortion from GoPro Hero 3+.')
    p.add_argument('path', help='Path containing JPG files.')
    return p.parse_args()

if __name__ == '__main__':
    args = get_args()
    file_list = []
    for root, sub_folders, files in os.walk(args.path):
        file_list += [os.path.join(root, filename) for filename in files if filename.lower().endswith(".jpg")]
    outdir = os.path.join(args.path, "undistort")
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    main(file_list, outdir)
