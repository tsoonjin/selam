#!/usr/bin/env python
import cv2
import numpy as np
import rospy

'''BGR COLOR CODES'''
RED = [0, 0, 255]
BLUE = [255, 128, 0]
YELLOW = [0, 255, 255]
GREEN = [0, 255, 0]
ORANGE = [0, 69, 255]
PURPLE = [204, 0, 204]
WHITE = [255, 255, 255]


class Color(object):

    @staticmethod
    def hue_to_bgr(hue):
        bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)
        return bgr[0][0].tolist()


class Logger(object):
    """ Enhance log with specific style to attract visual attention """

    BODY = '\033[{}m'
    ESC = '\033[0m'
    COLOR = {'black': '0', 'red': '1', 'green': '2', 'yellow': '3', 'blue': '4', 'magenta': '5',
             'cyan': '6', 'white': '7'}
    STYLE = {'dim': '1', 'bold': '2'}

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fail(self, content):
        rospy.loginfo(self.write_color("Fail: " + content, 'red'))

    def statesuccess(self, state, content):
        output = [""]
        output.append(self.write_color("[[Completed {} state]]".format(state).ljust(20),
                      'black', 'green', 'bold'))
        output.append(self.write_color(content.ljust(30), 'black', 'magenta', 'bold'))
        rospy.loginfo("\n".join(output))

    def taskoptions(self, content):
        output = [""]
        output.append(self.write_color("[[Task]]".ljust(15), 'black', 'red', 'bold'))
        for option, arg in content.items():
            option = "{}:".format(option)
            option = self.write_color((option).ljust(10), 'black', 'magenta', style='bold')
            arg = " {}".format(arg).ljust(30)
            arg = self.write_color(arg, 'black', 'magenta', style='bold')
            output.append(option + arg)
        rospy.loginfo("\n".join(output))

    def acousticsoptions(self, content):
        output = [""]
        output.append(self.write_color("[[Ping]]", 'black', 'cyan', 'bold'))
        output.append(self.write_color(content, 'black', 'magenta', 'bold'))
        rospy.loginfo("\n".join(output))

    def visionoptions(self, content):
        output = [""]
        output.append(self.write_color("[[VISION CONFIG]]", 'black', 'yellow', 'bold'))
        for obj_id, args in content.items():
            obj_id = "{}: ".format(obj_id)
            obj_id = self.write_color(obj_id.ljust(15), 'black', 'magenta', style='bold')
            args = " {}\n".format(args)
            args = self.write_color(args.ljust(20), 'black', 'magenta', style='bold')
            output.append(obj_id + args)
        rospy.loginfo("\n".join(output))

    def options(self, content):
        content = vars(content)
        output = [""]
        output.append(self.write_color("[[OPTIONS]]".ljust(15), 'black', 'yellow', 'bold'))
        for option, arg in content.items():
            option = "{}:".format(option)
            option = self.write_color((option).ljust(20), 'black', 'magenta', style='bold')
            arg = " {}".format(arg).ljust(20)
            arg = self.write_color(arg, 'black', 'magenta', style='bold')
            output.append(option + arg)
        rospy.loginfo("\n".join(output))

    def wait(self, content):
        rospy.loginfo(self.write_color("Waiting for: " + content, 'yellow'))

    def success(self, content):
        rospy.loginfo(self.write_color("Success: " + content, 'green'))

    def write_color(self, content, fg, bg=None, style=None):
        fg = '3' + Logger.COLOR[fg]
        bg = '4' + Logger.COLOR[bg] if bg else '49'
        body = Logger.BODY.format(fg + ';' + bg)
        body = body if not style else body + Logger.BODY.format(Logger.STYLE[style])
        return body + content + Logger.ESC
