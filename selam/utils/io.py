#!/usr/bin/env python
import os
import csv
import cv2

from multiprocessing import Pool


def write_img(img_name):
    """ Write img to given directory path
        @param img_name: (image, filename)
    """
    cv2.imwrite(img_name[1], img_name[0])


def write_to_dir(imgs, path, parallel=True):
    """ Write images to directory path
        @param imgs: images to be written
        @param path: output directory
        @param parallel: run using multiprocessing
    """
    img_names = [(i, '{}/{:08d}.jpg'.format(path, index)) for index, i in enumerate(imgs)]
    if parallel:
        pool = Pool()
        pool.map(write_img, img_names)
        pool.close()
        pool.join()


def get_basename(name):
    path = os.path.abspath(name)
    return os.path.basename(os.path.normpath(path))


def generate_csv(out_path, ls):
    """ Parse detected objects to csv file """
    with open(out_path, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(ls)
