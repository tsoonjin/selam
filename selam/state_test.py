#!/usr/bin/env python
import rospy
import actionlib
from bbauv_msgs.srv import Manipulators, ManipulatorsResponse, navigate2d, navigate2dResponse
from controls.msg import LocomotionAction
from std_msgs.msg._Float32 import Float32
from geometry_msgs.msg import Vector3Stamped


class Tester:
    """ Dummy Locomotion server for testing """
    def __init__(self): 
        self.server = actionlib.SimpleActionServer('LocomotionServer',
                                                   LocomotionAction, self.action_cb, False)
        self.data = {'heading': None, 'depth': None}

        # Initialize publishers
        self.depth_pub = rospy.Publisher("/depth", Float32)
        self.heading_pub = rospy.Publisher("/navigation/RPY", Vector3Stamped)

        # Initialize services
        self.manipulators_server = rospy.Service("/manipulators", Manipulators, self.mani_cb)
        self.nav2D_server = rospy.Service("/navigate2D", navigate2d, self.nav2d_cb)

    def action_cb(self, goal):
        rospy.logwarn("F: %.2f, SM: %.2f, D: %.2f, T: %.2f" %
                      (goal.forward_setpoint, goal.sidemove_setpoint, goal.depth_setpoint, goal.heading_setpoint))
        self.server.set_succeeded()

    def nav2d_cb(self, req):
        return navigate2dResponse(status=True)

    def mani_cb(self, req):
        rospy.logwarn(req)
        return ManipulatorsResponse(done=True)

if __name__ == "__main__":
    rospy.init_node("test_server")
    tester = Tester()
    rospy.logwarn("Starting Locomotion Server")
    tester.server.start()
    rospy.spin()
