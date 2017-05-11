#!/bin/bash
import os
import sys
import cv2


def get_roi(img, rect, size=(100, 100)):
    """ Return extracted bounding box given 4 corners of a rectangle
        size: size of training image
    """
    xpos = rect[0::2]
    ypos = rect[1::2]
    y = [int(min(ypos)), int(max(ypos))]
    x = [int(min(xpos)), int(max(xpos))]
    roi = img[y[0]:y[1], x[0]:x[1]]
    return cv2.resize(roi, size)


def get_jpgs(dirpath, skip=0, resize=None):
    """ Returns all images located in given dirpath
        skip : number of frames skip to reduce computation time
        resize: scale factor for resize

    """
    filenames = os.listdir(dirpath)
    # Only attempt to parse and sort files that end with .jpg
    filenames = [filename for filename in filenames
                 if filename.endswith(".jpg") or filename.endswith(".png")]
    filenames.sort(key=lambda x: int(x.split('.', 1)[0]))
    frames = [cv2.imread('{}/{}'.format(dirpath, filename))
              for filename in filenames]
    out = frames[0::skip] if skip > 0 else frames
    print('Read {} images from {}'.format(len(out), dirpath))
    if resize:
        new_size = (out[0].shape[1] / resize, out[0].shape[0] / resize)
        return map(lambda x: cv2.resize(x, new_size), out)
    return out


def extract_groundtruth(dataset_path, annotation):
    """ Returns a list of labelled images as positive training data """
    extracted = []
    imgs = get_jpgs(dataset_path)
    with open(annotation) as ann:
        for i, label in zip(imgs, ann):
            rect = map(float, label.rstrip().split(','))
            if rect[0] > 0:
                roi = get_roi(i, rect)
                extracted.append(roi)
    print("{} extracted".format(len(extracted)))
    return extracted


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python extract_region.py <dataset directory> <annotation file>\n")
        exit()
    extract_groundtruth(sys.argv[1], sys.argv[2])
