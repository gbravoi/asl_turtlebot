#!/usr/bin/env python

from enum import Enum

import rospy
from asl_turtlebot.msg import DetectedObject
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from std_msgs.msg import Float32MultiArray, String
from visualization_msgs.msg import Marker
import numpy as np
import tf



class Vendor:

    def __init__(self, position, name,marker_id):
        self.name=name
        self.position=position#tuple (x,y)
        self.marker_id=marker_id
        self.publisher= rospy.Publisher('vendor_marker/'+name, Marker, queue_size=10)
        colors = np.random.rand(1,3)
        print("COLORS: ", colors)
        self.marker_color= colors

    def publish_vendor_position(self):
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()

        # IMPORTANT: If you're creating multiple markers, 
        #            each need to have a separate marker ID.
        marker.id = self.marker_id

        marker.type = 2 # sphere

        marker.pose.position.x = self.position[0]
        marker.pose.position.y = self.position[1]
        marker.pose.position.z = 0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        marker.color.a = 1
        marker.color.r = self.marker_color[0][0]
        marker.color.g = self.marker_color[0][1]
        marker.color.b = self.marker_color[0][2]


        
        self.publisher.publish(marker)
    

class Mode(Enum):
    """State machine modes. Feel free to change."""
    IDLE = 1
    POSE = 2
    STOP = 3
    CROSS = 4
    EXPLORE = 5
    MANUAL = 6
    GO_TO_VENDOR=7
    WAIT_ON_VENDOR=8



class SupervisorParams:

    def __init__(self, verbose=False):
        # If sim is True (i.e. using gazebo), we want to subscribe to
        # /gazebo/model_states. Otherwise, we will use a TF lookup.
        self.use_gazebo = rospy.get_param("sim")

        # How is nav_cmd being decided -- human manually setting it, or rviz
        self.rviz = rospy.get_param("rviz")

        # If using gmapping, we will have a map frame. Otherwise, it will be odom frame.
        self.mapping = rospy.get_param("map")

        # Threshold at which we consider the robot at a location
        self.pos_eps = rospy.get_param("~pos_eps", 0.1)
        self.theta_eps = rospy.get_param("~theta_eps", 0.3)

        # Time to stop at a stop sign
        self.stop_time = rospy.get_param("~stop_time", 3.)

        # Minimum distance from a stop sign to obey it
        self.stop_min_dist = rospy.get_param("~stop_min_dist", 0.5)

        # Time taken to cross an intersection
        self.crossing_time = rospy.get_param("~crossing_time", 3.)

        if verbose:
            print("SupervisorParams:")
            print("    use_gazebo = {}".format(self.use_gazebo))
            print("    rviz = {}".format(self.rviz))
            print("    mapping = {}".format(self.mapping))
            print("    pos_eps, theta_eps = {}, {}".format(self.pos_eps, self.theta_eps))
            print("    stop_time, stop_min_dist, crossing_time = {}, {}, {}".format(self.stop_time, self.stop_min_dist, self.crossing_time))


class Supervisor:

    def __init__(self):
        # Initialize ROS node
        print("Supervisor Init")
        rospy.init_node('pavonecart', anonymous=True)
        self.params = SupervisorParams(verbose=True)

        # Current state
        self.x = 0
        self.y = 0
        self.theta = 0
        
        #points to explore map
        self.explore_points=[(3.5,0.5,0),(1.6,0.25,0), (0.25,0.25,0),(0.8,2.8,0),(1.6,2.8,0),(1.6,1.6,0)]


        # Goal state
        self.x_g = 0
        self.y_g = 0
        self.theta_g = 0

        # Current mode
        self.mode = Mode.IDLE
        self.prev_mode = None  # For printing purposes

        self.vendors_to_visit = []
        self.vendor_dic = {}  #list of all vendors
        self.current_vendor_index=0
        ########## PUBLISHERS ##########

        # Command pose for controller
        self.pose_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)

        # Command vel (used for idling)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        ########## SUBSCRIBERS ##########
        #request subcriber
        rospy.Subscriber('/delivery_request', String, self.delivery_request_callback)


        # Stop sign detector
        # rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)

        # High-level navigation pose
        #rospy.Subscriber('/cmd_nav', Pose2D, self.nav_pose_callback)

        # If using gazebo, we have access to perfect state
        if self.params.use_gazebo:
            rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_callback)
        self.trans_listener = tf.TransformListener()

        # If using rviz, we can subscribe to nav goal click
        if self.params.rviz:
            rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)
        else:
            self.x_g, self.y_g, self.theta_g = 1.5, -4., 0.
            self.mode = Mode.EXPLORE
        

    ########## SUBSCRIBER CALLBACKS ##########
    def delivery_request_callback(self,msg):
        stores = msg.data.split(",") #[banana, apple]
        print('stores:', stores)
        for vendor in stores:
            if vendor in self.vendor_dic and vendor not in self.vendors_to_visit:
                self.vendors_to_visit.append(vendor)
        self.init_go_to_vendor()
    
    def gazebo_callback(self, msg):
        if "turtlebot3_burger" not in msg.name:
            return

        pose = msg.pose[msg.name.index("turtlebot3_burger")]
        self.x = pose.position.x
        self.y = pose.position.y
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]

    def rviz_goal_callback(self, msg):
        """ callback for a pose goal sent through rviz """
        origin_frame = "/map" if self.params.mapping else "/odom"
        print("Rviz command received!")

        try:
            nav_pose_origin = self.trans_listener.transformPose(origin_frame, msg)
            self.x_g = nav_pose_origin.pose.position.x
            self.y_g = nav_pose_origin.pose.position.y
            quaternion = (nav_pose_origin.pose.orientation.x,
                          nav_pose_origin.pose.orientation.y,
                          nav_pose_origin.pose.orientation.z,
                          nav_pose_origin.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.theta_g = euler[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        self.mode = Mode.EXPLORE  ## CHANGE LATER TO CLICK 

    # def nav_pose_callback(self, msg):
    #     self.x_g = msg.x
    #     self.y_g = msg.y
    #     self.theta_g = msg.theta
    #     self.mode = Mode.NAV

    # def stop_sign_detected_callback(self, msg):
    #     """ callback for when the detector has found a stop sign. Note that
    #     a distance of 0 can mean that the lidar did not pickup the stop sign at all """

    #     # distance of the stop sign
    #     dist = msg.distance

    #     # if close enough and in nav mode, stop
    #     if dist > 0 and dist < self.params.stop_min_dist and self.mode == Mode.NAV:
    #         self.init_stop_sign()


    ########## STATE MACHINE ACTIONS ##########

    ########## Code starts here ##########
    # Feel free to change the code here. You may or may not find these functions
    # useful. There is no single "correct implementation".

    def init_go_to_vendor(self):
        if self.mode==Mode.IDLE:
            #next vendor name
            self.current_vendor_index=0
            if self.current_vendor_index<len(self.vendors_to_visit):
                self.mode=Mode.GO_TO_VENDOR
                vendor_name=self.vendors_to_visit[self.current_vendor_index]
                print("vendor_name ",vendor_name)
                vendor=self.vendor_dic[vendor_name]
                #vendor position
                self.x_g=vendor.position[0]
                self.y_g=vendor.position[1]
                self.theta_g=0
                print("vendor_position x:{} y:{} th:{}".format(self.x_g,self.y_g,self.theta_g))



    def go_to_vendor(self):
        self.current_vendor_index+=1

        if self.current_vendor_index<len(self.vendors_to_visit):
            vendor_name=self.vendors_to_visit[self.current_vendor_index]
            print("vendor_name ",vendor_name)
            vendor=self.vendor_dic[vendor_name]
            #vendor position
            self.x_g=vendor.position[0]
            self.y_g=vendor.position[1]
            self.theta_g=0
            print("vendor_position x:{} y:{} th:{}".format(self.x_g,self.y_g,self.theta_g))
            return True
        else: #we dont have more vendors to go
            return False



    def go_to_pose(self,point):
        """ sends the current desired pose to the pose controller """
        print("desires pose"+str(point))
        pose_g_msg = Pose2D()
        pose_g_msg.x =point[0] #
        pose_g_msg.y =point[1]#
        pose_g_msg.theta =  point[2]#

        self.x_g=point[0]
        self.y_g=point[1]
        self.theta_g=point[2]

        self.pose_goal_publisher.publish(pose_g_msg)

    def nav_to_pose(self):
        """ sends the current desired pose to the navigator """

        nav_g_msg = Pose2D()
        nav_g_msg.x = self.x_g
        nav_g_msg.y = self.y_g
        nav_g_msg.theta = self.theta_g

        self.pose_goal_publisher.publish(nav_g_msg)

    def stay_idle(self):
        """ sends zero velocity to stay put """

        vel_g_msg = Twist()
        vel_g_msg.linear.x = 0
        vel_g_msg.linear.y = 0
        vel_g_msg.linear.z = 0
        vel_g_msg.angular.x = 0
        vel_g_msg.angular.y = 0
        vel_g_msg.angular.z = 0
        self.cmd_vel_publisher.publish(vel_g_msg)

    def close_to(self, x, y, theta):
        """ checks if the robot is at a pose within some threshold """

        return abs(x - self.x) < self.params.pos_eps and \
               abs(y - self.y) < self.params.pos_eps
               # and \ abs(theta - self.theta) < self.params.theta_eps

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """
        if self.mode != Mode.CROSS:
            self.stop_sign_start = rospy.get_rostime()
            self.mode = Mode.STOP

    def init_wait_on_vendor(self):
        """ initiates wait on vendor """
        self.wait_on_vendor_start = rospy.get_rostime()
        self.mode = Mode.WAIT_ON_VENDOR

    def has_stopped(self):
        """ checks if stop sign maneuver is over """

        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.params.stop_time)

    def has_stopped_on_vendor(self):
        """ checks if waiting on vendor is over """

        return self.mode == Mode.WAIT_ON_VENDOR and \
               rospy.get_rostime() - self.wait_on_vendor_start > rospy.Duration.from_sec(self.params.stop_time)

    def init_crossing(self):
        """ initiates an intersection crossing maneuver """

        self.cross_start = rospy.get_rostime()
        self.mode = Mode.CROSS

    def has_crossed(self):
        """ checks if crossing maneuver is over """

        return self.mode == Mode.POSE and \
               rospy.get_rostime() - self.cross_start > rospy.Duration.from_sec(self.params.crossing_time)

    ########## Code ends here ##########


    ########## STATE MACHINE LOOP ##########

    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """

        if not self.params.use_gazebo:
            try:
                origin_frame = "/map" if self.params.mapping else "/odom"
                translation, rotation = self.trans_listener.lookupTransform(origin_frame, '/base_footprint', rospy.Time(0))
                self.x, self.y = translation[0], translation[1]
                self.theta = tf.transformations.euler_from_quaternion(rotation)[2]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass

        # logs the current mode
        if self.prev_mode != self.mode:
            rospy.loginfo("Current mode: %s", self.mode)
            self.prev_mode = self.mode

        ########## Code starts here ##########
        # TODO: Currently the state machine will just go to the pose without stopping
        #       at the stop sign.

        if self.mode == Mode.IDLE:
            # Send zero velocity
            print("IDLE")
            self.stay_idle()
            
        # elif self.mode == Mode.POSE:
        #     print("POSE")
        #     if self.has_crossed():
        #         self.mode=Mode.NAV
        #     else:
        #         # Moving towards a desired pose
        #         if self.close_to(self.x_g, self.y_g, self.theta_g):
        #             self.mode = Mode.IDLE
        #         else:
        #             self.go_to_pose()

        # elif self.mode == Mode.STOP:
        #     print("STOP")
        #     # At a stop sign
        #     if self.has_stopped():
        #         self.init_crossing()
        #     else:
        #         self.stay_idle()

        # elif self.mode == Mode.CROSS:
        #     print("CROSS")
        #     # Crossing an intersection
        #     self.mode=Mode.POSE

        elif self.mode == Mode.EXPLORE:
            print("EXPLORE")
            if self.close_to(self.x_g,self.y_g,self.theta_g):
                if len(self.explore_points)>0:
                    point = self.explore_points.pop(0)
                    self.go_to_pose(point)
                else:
                    self.mode = Mode.IDLE
            self.go_to_pose((self.x_g,self.y_g,self.theta_g))

        elif self.mode==Mode.GO_TO_VENDOR:
            print("GO_TO_VENDOR")
            self.go_to_pose((self.x_g,self.y_g,self.theta_g))
            if self.close_to(self.x_g,self.y_g,self.theta_g):
                self.init_wait_on_vendor()



        elif self.mode==Mode.WAIT_ON_VENDOR:
            #WAITING TIME
            if self.has_stopped_on_vendor():
                #WHEN TIME IS OVER
                if self.go_to_vendor():
                    self.mode=Mode.GO_TO_VENDOR
                else:
                    #clean list of visited vendor
                    self.vendors_to_visit=[]
                    self.mode = Mode.IDLE #change leter

        else:
            raise Exception("This mode is not supported: {}".format(str(self.mode)))

        ############ Code ends here ############

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        #set first point as goal
        point = self.explore_points.pop(0)
        self.go_to_pose(point)
        while not rospy.is_shutdown():
            self.loop()

            for vendor in self.vendor_dic.values():
                #print(vendor.name)
                vendor.publish_vendor_position()

            rate.sleep()


if __name__ == '__main__':

    #creating vendors, delte later
    vendor1=Vendor((0.5,0.2), "apple",0)
    vendor2=Vendor((1.6,0.2), "banana",1)

    sup = Supervisor()
    sup.vendor_dic[vendor1.name] = vendor1#delet later
    sup.vendor_dic[vendor2.name] = vendor2
    sup.run()
