#!/usr/bin/env python
import cv2


class Tracker(object):
    """ Generic tracker class """
    def __init__(self, imgs, detector):
        self.detector = detector
        self.imgs = imgs
        self.detected = []

    def generate_result(self):
        """ Runs detector algorithm on images """
        for frame in self.imgs:
            filename = frame[0].split('.')[0]
            raw_img = frame[1]

    def process_img(self, img):
        pass

