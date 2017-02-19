#!/usr/bin/env python
""" Annotates region of interest by initializing a bounding box using MeanShift """
import cv2
import numpy as np
from selam.utils import img
from selam.enhancement import shadegrey


drawing = False
box = None


def mouse_cb(event, x, y, flags, params):
    global drawing, box
    if event == cv2.EVENT_LBUTTONDOWN:
        box = (x, y)
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        box = (x, y)
        drawing = False


def main():
    path = './examples/dataset/robosub16/FRONT/0-264_buoys'
    imgs = img.get_jpgs(path)
    init_frame = shadegrey(imgs[0])
    cv2.namedWindow('meanshift')
    cv2.setMouseCallback('meanshift', mouse_cb)
    cv2.imshow('meanshift', init_frame)
    # @TODO let user confirms bounding box before moving on
    # Wait 3 seconds for user to select bounding box
    cv2.waitKey(3000)

    while not box:
        pass
    # Setup initial location of window
    cv2.circle(init_frame, box, 10, [0, 255, 0], 2)
    cv2.rectangle(init_frame, box, (box[0] + 70, box[1] + 70), 255, 2)
    cv2.imshow('meanshift', init_frame)
    cv2.waitKey(3000)
    x, y, w, h = [box[0], box[1], 60, 60]
    track_window = (x, y, w, h)
    roi = cv2.cvtColor(shadegrey(imgs[0])[y: y + h, x: x + w], cv2.COLOR_BGR2LAB)
    roi_hist = cv2.calcHist([roi], [0], None, [255], [0, 255])
    roi_hist = cv2.normalize(roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1)
    for f in imgs[1:]:
        f = shadegrey(f)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        x, y, w, h = track_window
        img2 = cv2.rectangle(f, (x, y), (x+w, y+h), 255, 2)
        cv2.imshow('meanshift', img2)
        cv2.waitKey(1000)
if __name__ == '__main__':
    main()
