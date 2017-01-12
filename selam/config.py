#!/usr/bin/python2.7
import numpy as np

# ROS
from std_msgs.msg._Float32 import Float32
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from sensor_msgs.msg import Image
from vision.srv import Detector
from controls.msg import LocomotionAction
from controls.srv import Controller
from acoustics.msg import Ping
from bbauv_msgs.srv import navigate2d


class BottomTask(object):
    grabber_offset = 0.4


class Config(object):
    """ Task specific configs """

    # List of subscribeable topics
    topics = {'heading': ('/navigation/RPY', Vector3Stamped),
              'depth': ('/depth', Float32),
              'frontcam': ('/front_camera/camera/image_color', Image),
              'bottomcam': ('/bot_camera/camera/image_color', Image),
              'vision_pipeline': ('/vision/detector', Detector),
              'nav_server': ('LocomotionServer', LocomotionAction),
              'controller': ('/controller', Controller),
              'navigate2D': ('/navigate2D', navigate2d),
              'earth_odom': ('/navigation/odom/earth', PoseStamped),
              'acoustics': ('/acoustics/ping', Ping)}

    # List of reconfigurable nodes

    reconf_nodes = {'bot_camera': 'bot_camera/bot_camera',
                    'front_camera': 'front_camera/front_camera'}

    # Obstacles information

    length = {'cover': 0.6, 'bin': 0.6, 'overall_bin': 0.6,
              'coin': 0.15, 'horizontal_coin': 0.15, 'vertical_coin': 0.025, 'red_coin': 0.025,
              'green_coin': 0.025, 'xmark': 0.225, 'table': 1.2}

    depth = {'cover': 4.0, 'bin': 4.0, 'overall_bin': 4.0,
             'coin': 3.0, 'horizontal_coin': 3.0, 'vertical_coin': 3.0, 'red_coin': 3.0, 'green_coin': 3.0,
             'xmark': 4.0, 'table': 4.0}

    ''' Camera '''

    processed_img_size = (320, 240)
    bot_center = [160, 120]
    # For 1280x960 resolution obtained using MatLab world unit 'mm'
    bot_camera_matrix = np.matrix([
                        [1343.595179, 0.000000000, 630.0921],
                        [0.000000000, 1344.026410, 445.8840],
                        [0.000000000, 0.000000000, 1.000000]
                        ])
    bot_focal_length = [1343.595179, 1344.026410]  # Multiplied with 1.33 in water
    bot_principal_point = [630.0921, 445.8840]
    bot_distortion_coeff = [-0.2129, 0.1240, 0., 0., 0.]
    bot_pixel_size = [0.000004, 0.000004]
    bot_resize_factor = 4
    # Using 320 x 240 resolution 76.7 (fov) / resolution size. In radian/pixel
    bot_angular_resolution = [0.00418, 0.00558]
    # bot_distance_factor = 0.00566  # Using 320 x 240 resolution with bin cover as reference in meter

    def __init__(self, name):
        self.name = name
