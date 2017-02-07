#!/usr/bin/env python
""" Generates result for validation from a dummy detector """
import json
import random
from selam.utils import img, io
from selam.detector import detector


class DummyDetector(detector.Detector):
    def __init__(self, target_ids):
        super(DummyDetector, self).__init__(target_ids)

    def detect(self, img):
        """ Only detects the first target id """
        res = {}
        res['id'] = self.target_ids[0]
        res['width'], res['height'] = 300, 400
        y_bound, x_bound = img.shape[:2]
        res['x'] = random.uniform(0.0, x_bound - res['width'] - 1)
        res['y'] = random.uniform(0.0, y_bound - res['height'] - 1)
        res['conf'] = random.uniform(0.0, 1.0)
        return res


def main():
    detector = DummyDetector([1])
    path = './examples/dataset/robosub16/FRONT/0-264_buoys'
    det_file = './examples/output/det/det.txt'
    res_file = './examples/output/res/0-264_buoys.txt'
    imgs = img.get_jpgs(path)
    detected = []
    tracked = []

    for i, frame in enumerate(imgs):
        res = detector.detect(frame)
        detected.append([i + 1, -1, res['x'], res['y'], res['width'],
                         res['height'], res['conf'], -1, -1, -1])
        tracked.append([i + 1, detector.target_ids[0], res['x'], res['y'],
                        res['width'], res['height'], -1, -1, -1, -1])

    io.generate_csv(det_file, detected)
    io.generate_csv(res_file, tracked)


if __name__ == '__main__':
    main()
