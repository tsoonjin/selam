#!/usr/bin/python2.7
import rospy
from std_msgs.msg import Header
from vision.msg import DetectedObject


class Frame(object):
    """ Unit frame of image along with data about detected objects """
    def __init__(self, img):
        self.header = Header(stamp=rospy.Time.now())
        self.img = img
        self.__detected_objects = {'bin': DetectedObject(name='bin'),
                                   'cover': DetectedObject(name='cover'),
                                   'overall_bin': DetectedObject(name='overall_bin'),
                                   'green_coin': DetectedObject(name='green_coin'),
                                   'red_coin': DetectedObject(name='red_coin'),
                                   'vertical_coin': DetectedObject(name='vertical_coin'),
                                   'horizontal_coin': DetectedObject(name='horizontal_coin'),
                                   'tower': DetectedObject(name='tower'),
                                   'xmark': DetectedObject(name='xmark'),
                                   'green_x': DetectedObject(name='green_x'),
                                   'red_x': DetectedObject(name='red_x'),
                                   'generic': DetectedObject(name='generic'),
                                   'table': DetectedObject(name='table')}

    def __repr__(self):
        return "Frame {}: Detected: {}".format(self.header.stamp, self._get_detected_objects())

    def get_object(self, obj_id):
        return self.__detected_objects[obj_id]

    def add_object(self, info):
        self.__detected_objects[info.name] = info

    def _get_detected_objects(self):
        return [obj_id for obj_id, info in self.__detected_objects.items() if info.detected]
