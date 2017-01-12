#!/usr/bin/python2.7
import time
import signal
import rospy
import actionlib

from dynamic_reconfigure.client import Client
from vision.utils.constant import Logger
from vision.srv import DetectorRequest
from controls.msg import LocomotionGoal
from bbauv_msgs.srv import (mission_to_vision, mission_to_visionResponse,
                            vision_to_mission)
from bbauv_msgs.msg import controller


class Comm(object):
    """Main communication class that interacts with main system"""

    def __init__(self, config):
        self.topics = config.topics
        self.config = config
        signal.signal(signal.SIGINT, self.handle_SIGINT)

        """ State of a task
            is_activated:  true when activated by mission planner
            is_preempted:  true when task is preempted by user or mission planner
            is_registered: true when task has initialized all components
            is_test:       true when testing a particular state
            is_alone:      true when running task independent of mission planner
            is_done:       true when a task is completed
        """
        self.logger = Logger()
        self.state = {'is_activated': False, 'is_preempted': False, 'is_registered': False,
                      'starting_state': rospy.get_param('~state', 'started'),
                      'is_test': rospy.get_param('~test', False),
                      'is_alone': rospy.get_param('~alone', False),
                      'is_done': False}

        self.data = {'heading': None, 'depth': None, 'start_time': time.time(), 'pos2D': None}
        self.camera_client = self.init_camera_client()
        self.subscribers = []
        self.publishers = {}
        self.vision_server_topic = "/{}/mission_to_vision".format(config.name)
        self.mission_server_topic = "/{}/vision_to_mission".format(config.name)
        if not self.state['is_alone']:
            self.vision_server = rospy.Service(self.vision_server_topic, mission_to_vision,
                                               self.mission_request_cb)

        # Weight for movement differential
        self.w_dx = 1.5
        self.w_dy = 0.5
        self.move_timeout = 10
        self.abort_timeout = 30

    def init_camera_client(self):
        try:
            camera_client = Client(self.config.reconf_nodes['bot_camera'], timeout=1,
                                   config_callback=self.camera_reconf_cb)
            return camera_client
        except Exception as e:
            self.logger.fail(str(e))
            return None

    def update_shutter(self, shutter_val):
        if self.camera_client:
            self.camera_client.update_configuration({'shutter': shutter_val})

    def init_subs(self):
        self.subscribers.append(rospy.Subscriber(self.topics['heading'][0],
                                                 self.topics['heading'][1], self.heading_cb))
        self.subscribers.append(rospy.Subscriber(self.topics['depth'][0],
                                                 self.topics['depth'][1], self.depth_cb))
        self.subscribers.append(rospy.Subscriber(self.topics['earth_odom'][0],
                                                 self.topics['earth_odom'][1], self.earth_cb))
        self.init_extra_subs()
        rospy.loginfo('Completed subscriptions')

    def init_extra_subs(self):
        """ To be implemented to include more subscribers """
        pass

    def get_subscribers(self):
        return [sub.name for sub in self.subscribers]

    def init_services(self):
        self.nav2D_client = rospy.ServiceProxy(self.config.topics['navigate2D'][0],
                                               self.config.topics['navigate2D'][1])
        self.init_extra_srvs()
        rospy.loginfo('Completed services')

    def init_extra_srvs(self):
        """ To be implemented to include more services """
        pass

    def init_navigation(self):
        """Connect with action server in charge of vehicle navigation"""
        self.ac = actionlib.SimpleActionClient(self.topics['nav_server'][0],
                                               self.topics['nav_server'][1])
        try:
            self.ac.wait_for_server()
        except:
            rospy.loginfo("LocomotionServer timeout")

    def init_mission_bridge(self):
        """ Initializes a service client to mission planner """
        if not self.state['is_alone']:
            self.mission_bridge = rospy.ServiceProxy(self.mission_server_topic, vision_to_mission)
            self.mission_bridge.wait_for_service()

    def init_controller(self):
        self.controller = rospy.ServiceProxy(self.topics['controller'][0],
                                             self.topics['controller'][1])
        rospy.loginfo("Waiting for controller")

    def toggle_controller(self, toggle):
        self.controller(toggle)

    def init_all(self):
        if not self.state['is_registered']:
            self.init_subs()
            self.init_services()
            self.init_controller()
            self.init_navigation()
            self.init_mission_bridge()
            self.state['is_registered'] = True
        self.data['start_time'] = time.time()
        self.wait_for_sensors()

    def wait_for_sensors(self):
        """ Wait for sensors to be populated before running pipeline """
        rospy.loginfo("Waiting for sensors to be populated")
        while not self.data['depth']:
            rospy.sleep(rospy.Duration(1.0))
        while not self.data['heading']:
            rospy.sleep(rospy.Duration(1.0))
        rospy.loginfo("All sensors populated")

    def heading_cb(self, data):
        self.data['heading'] = data.vector.z

    def depth_cb(self, data):
        self.data['depth'] = data.data

    def earth_cb(self, data):
        self.data['pos2D'] = data

    def camera_reconf_cb(self, config):
        pass

    def handle_SIGINT(self, signal, frame):
        self.state['is_preempted'] = True

    def mission_request_cb(self, req):
        if req.start_request:
            rospy.logwarn("Started")
            self.state['is_activated'] = True
            return mission_to_visionResponse(start_response=True,
                                             abort_response=False)
        elif req.abort_request:
            rospy.logwarn("Aborted")
            self.state['is_preempted'] = True
            self.state['is_activated'] = False
            return mission_to_visionResponse(start_response=False,
                                             abort_response=True)

    def send_task_abort(self):
        """Signal mission planner a task is aborted"""
        if not self.state['is_alone']:
            self.mission_bridge(fail_request=True, task_complete_request=False,
                                task_complete_ctrl=controller(heading_setpoint=self.data['heading']))
        time_elapsed = time.time() - self.data['start_time']
        self.logger.fail("Aborted {0} in {1:.2f}s: Heading: {2:.2f}, Depth: {3:.2f}".format(
            self.config.name, time_elapsed, self.data['heading'], self.data['depth']))

    def send_task_complete(self):
        """Signal mission planner a task is completed"""
        if not self.state['is_alone']:
            self.mission_bridge(fail_request=False, task_complete_request=True,
                                task_complete_ctrl=controller(heading_setpoint=self.data['heading']))
        self.state['is_activated'] = False
        time_elapsed = time.time() - self.data['start_time']
        self.logger.success("Completed {0} in {1:.2f}s: Heading: {2:.2f}, Depth: {3:.2f}".format(
            self.config.name, time_elapsed, self.data['heading'], self.data['depth']))

    """ Manipulators: top_torpedo, bot_torpedo, dropper, grabber, rotary_vertical, rotary_horizontal """

    def open_arm(self):
        res = self.mani_req(False, False, False, True, True, False)
        rospy.loginfo("Manipulator {}".format(res))

    def close_arm(self):
        res = self.mani_req(False, False, False, False, True, False)
        rospy.loginfo("Manipulator {}".format(res))

    def retract_arm(self):
        res = self.mani_req(False, False, False, False, False, True)
        rospy.loginfo("Manipulator {}".format(res))

    def lower_arm(self):
        res = self.mani_req(False, False, False, False, True, False)
        rospy.loginfo("Manipulator {}".format(res))

    def drop(self):
        res = self.mani_req(False, False, True, False, False, False)
        rospy.loginfo("Manipulator {}".format(res))

    ''' Movement & State Machine  '''

    def move(self, forward=0.0, sidemove=0.0, turn=0.0, depth=None, relative=True, duration=None):
        depth = depth if depth else self.data['depth']
        turn = (self.data['heading'] + turn) % 360 if relative else turn
        goal = LocomotionGoal(forward_setpoint=forward, heading_setpoint=turn,
                              sidemove_setpoint=sidemove, depth_setpoint=depth)
        if self.options.verbose:
            rospy.loginfo("F:%.2f, SM:%.2f, T:%.2f, D:%.2f" % (forward, sidemove, turn, depth))
        self.ac.send_goal(goal)

        if duration:
            self.ac.wait_for_result(rospy.Duration(duration))
        else:
            self.ac.wait_for_result(rospy.Duration(self.move_timeout))
            rospy.logwarn("Timeout waiting for response from LocomotionServer")

    def async_move(self, forward=0.0, sidemove=0.0, turn=0.0, depth=None, relative=True):
        depth = depth if depth else self.data['depth']
        turn = (self.data['heading'] + turn) % 360 if relative else turn
        goal = LocomotionGoal(forward_setpoint=forward, heading_setpoint=turn,
                              sidemove_setpoint=sidemove, depth_setpoint=depth)
        rospy.loginfo("Async F: %f, SM: %f, T: %f, D: %f" % (forward, sidemove, turn, depth))
        self.ac.send_goal(goal)

    def diff_centering(self, state, obj_id, outcome, center_depth, limit=0.05, turn=True):
        """ Moves towards centroid of detected object via differential in pixel distance
            state_transition    SMACH outcome after centering to object
            center_depth        vehicle's depth while centering
            limit               criteria for successful centering """
        # Initialization
        start_time = time.time()
        explore = False

        while not self.state['is_preempted']:
            target = self.target[obj_id]

            if target and target.detected:
                dx = target.offset[0]
                dy = target.offset[1]
                if abs(dx) <= limit and abs(dy) <= limit:
                    self.logger.success('Centered to {} turning {}'.format(target.name,
                                                                           target.angle))
                    self.state['color'] = target.predicted_color
                    if turn:
                        self.move(turn=target.angle, depth=center_depth)
                    return self.complete_state(state, outcome)
                self.move(forward=dy * self.w_dy, sidemove=dx * self.w_dx, depth=center_depth,
                          duration=0.3)
            elif (time.time() - start_time) > self.move_timeout and not explore:
                self.search_object(obj_id)
                explore = True
            elif (time.time() - start_time) > self.abort_timeout:
                self.send_task_abort()
                return 'aborted'
        return 'aborted'

    def diff_diving(self, obj_id, depth_limit, depth_offset=0.1, limit=0.1, turn=False):
        """ Dive towards centroid of detected object via differential in pixel distance
            state_transition    SMACH outcome after centering to object
            depth_limit         depth at which the action will be completed
            depth_offset        depth increase for diving
            limit               criteria for successful centering """
        # Initialization
        start_time = time.time()
        explore = False

        while not self.state['is_preempted']:
            target = self.target[obj_id]

            if target and target.detected:
                dx = target.offset[0]
                dy = target.offset[1]
                if self.data['depth'] >= depth_limit and abs(dx) <= limit and abs(dy) <= limit:
                    self.logger.success('Centered to {} turning {}'.format(target.name,
                                                                           target.angle))
                    if turn:
                        self.move(turn=target.angle)
                    return
                self.w_dy
                center_depth = self.data['depth'] + depth_offset
                self.move(forward=dy * self.w_dy, sidemove=dx * self.w_dx, depth=center_depth,
                          duration=0.1)
            elif (time.time() - start_time) > self.move_timeout and not explore:
                self.search_object(obj_id)
                explore = True
            elif (time.time() - start_time) > self.abort_timeout:
                self.send_task_abort()
                return 'aborted'
        return 'aborted'

    def setpoint_centering(self, obj_id, outcome, center_depth=None):
        """ Moves towards centroid of object via setpoint
            state_transition    SMACH outcome after centering to object
            center_depth        vehicle's depth while centering """

        while not self.state['is_preempted']:
            target = self.target[obj_id]
            if target and target.detected:
                self.move(forward=target.move[1], sidemove=target.move[0], depth=center_depth)
        return 'aborted'

    def start_tracking(self, target):
        req = DetectorRequest()
        setattr(req, target, True)
        res = self.vision_client(req)
        rospy.logwarn("Tracking {}: {}".format(target, res.done))

    def complete_state(self, state, outcome):
        time_elapsed = time.time() - state.start_time
        self.logger.statesuccess(state.__class__.__name__,
                                 "Time taken {0:.2f}s: Heading: {1:.2f}, Depth: {2:.2f}".format(
                                  time_elapsed, self.data['heading'], self.data['depth']))
        return outcome

    def move2D(self, obj_id, outcome):
        try:
            res = self.nav2D_client(*self.state[obj_id])
            self.logger.success(str(res.status))
            return outcome
        except Exception as e:
            rospy.logwarn("Failed to move: {}".format(e))
            return 'aborted'

    def log_pos2D(self, obj_id):
        """ Log current x, y, heading """
        if self.data['pos2D']:
            self.state[obj_id] = [self.data['pos2D'].pose.position.x,
                                  self.data['pos2D'].pose.position.y,
                                  self.data['heading']]
            self.logger.success('Recorded {} at: X: {}, Y: {}, HEADING: {}'.format(obj_id,
                                self.state[obj_id][0], self.state[obj_id][1], self.state[obj_id][2]))

    def search_object(self, obj_id, search_dist=1.5, detected_limit=3):
        self.logger.wait("Searching for {}".format(obj_id))
        next_move = 0
        steps = [(-search_dist, 0), (search_dist, search_dist), (search_dist, -search_dist),
                 (-search_dist, -search_dist)]
        while not self.state['is_preempted']:
            target = self.target[obj_id]
            if target and target.detected:
                break
            else:
                self.move(forward=steps[next_move][1], sidemove=steps[next_move][0], duration=7.0)
                next_move = (next_move + 1) % len(steps)
                target = self.target[obj_id]
