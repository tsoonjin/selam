#!/usr/bin/env python
""" Explore different object proposals algorithm """
import cv2
from lib import ssearch
from selam.utils import img
from selam.enhancement import shadegrey


def mserSearch(img):
    """ Using different thresholds on grayscale image to detect different components """
    mser = cv2.MSER_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(img, hulls, 1, (0, 255, 0))
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def selective_search(img):
    """ Performs selective search and draw all detected regions """
    labels, regions = ssearch.selective_search(img, scale=10, sigma=0.9, min_size=10)
    for r in regions:
        box = r['rect']
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow('main', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def main():
    path = './examples/dataset/robosub16/SEMI/4142-4424_tower_bright'
    imgs = img.get_jpgs(path, resize=2)
    chosen = shadegrey(imgs[31])
    mserSearch(chosen)


if __name__ == '__main__':
    main()
