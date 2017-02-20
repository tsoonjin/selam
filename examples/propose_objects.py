#!/usr/bin/env python
""" Explore different object proposals algorithm """
import cv2
from lib import ssearch
from selam.utils import img
from selam.enhancement import shadegrey


def selective_search(img):
    """ Performs selective search and draw all detected regions """
    labels, regions = ssearch.selective_search(img, scale=10, sigma=0.9, min_size=10)
    for r in regions:
        box = r['rect']
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow('main', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def main():
    path = './examples/dataset/robosub16/SEMI/677-1226_tower_table'
    imgs = img.get_jpgs(path, resize=2)
    imgs = map(lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2RGB), imgs)
    chosen = shadegrey(imgs[0])


if __name__ == '__main__':
    main()
