#!/usr/bin/env python


class Detector(object):
    """ Generic detector class """
    def __init__(self, target_ids):
        # Object ids to be detected
        self.target_ids = target_ids

    def detect(self, img):
        """ Returns dict with following properties:
            id: detected object id
            processed_frame: img with annotations and extra info
            x: leftmost x-coordinate
            y: topmost y-coordinate
            width: width of bounding box
            height: height of bounding box
            conf: confidence score
        """
        pass
