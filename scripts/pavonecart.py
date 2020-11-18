#!/usr/bin/env python

from enum import Enum

import rospy
from asl_turtlebot.msg import DetectedObject
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from std_msgs.msg import Float32MultiArray, String, Bool
from asl_turtlebot.msg import DetectedObject, DetectedObjectList
from visualization_msgs.msg import Marker
import numpy as np
import tf
import utils



class Vendor:

    def __init__(self, position, name,marker_id,distance_detected):
        self.name=name
        self.position=position#tuple (x,y)
        self.distance_detected=distance_detected
        self.marker_id=marker_id
        self.publisher= rospy.Publisher('vendor_marker/'+name, Marker, queue_size=10)
        colors = np.random.rand(1,3)
        print("Vendor {} created in position {}".format(self.name,self.position))
        self.marker_color= colors

    def publish_vendor_position(self):
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()

        # IMPORTANT: If you're creating multiple markers, 
        #            each need to have a separate marker ID.
        marker.id = self.marker_id

        marker.type = 1# sphere
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.a = 1
        marker.color.r = self.marker_color[0][0]
        marker.color.g = self.marker_color[0][1]
        marker.color.b = self.marker_color[0][2]


        if self.name=="dog":
            marker.type = 9
            marker.text="Dog"
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 1
            marker.color.g = 1
            marker.color.b = 1

        marker.pose.position.x = self.position[0]
        marker.pose.position.y = self.position[1]
        marker.pose.position.z = 0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0






        
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
    DELIVER=9
    STUCK=10
    WAYPOINT = 11



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
        self.pos_eps = rospy.get_param("~pos_eps", 0.3)
        
        self.theta_eps = rospy.get_param("~theta_eps", 0.3)

        # Time to stop at a stop sign
        self.stop_time = rospy.get_param("~stop_time", 3.)
        self.vendor_stop_time = 10

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
        self.x = 3.15
        self.y = 1.6
        self.theta = 0

        self.initial_pos=(3.15,1.6,0)
        
        #points to explore map
        # self.explore_points=[
        #     (3.3,2.8,0),
        #     (3.3,1.4,0),
        #     (3.1,0.4,0),
        #     (1.6,0.25,0), 
        #     (0.25,0.25,0),
        #     (0.8,2.8,0),
        #     (0.25,1.5,0),
        #     (2.4,1.8,0),
        #     (1.5,2.8,0),
        #     (1.5,1.5,0)]

        self.explore_points=[
            (3.1, 0.4, 1.57), #by pizza
            (2.5, 0.4, 0), #by banana
            
            (0.3, 0.3, 0), #by origin
            (0.3, 1.5, 1.57), #by apple #We need to stop by apple
            (0.6, 2.7, 0), #by curved corner 
            (2.3, 2.8, 0), #by sandwich
            (1.5, 2.7, 0), 
            (2.2, 1.6, 0), #by orange 
            (1.5, 1.5, 0),
        ]

        #waypoints for getting unstuck
        self.way_points = [
            (2.4,0.3,0),
            (2.4,2.7,0),
            (2.4,1.5,0),
            (0.3,1.5,0)
        ]

        self.remaining_way_points = []

        # Goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None
        self.previous_goal=[0,0,0] #if get stuck, save her previous goal
        self.previous_pos=np.array([-100,-100,-100])
        self.previous_mode=None #mode before getting stuck

        # Current mode
        self.mode = Mode.IDLE #start IDLE by exploring using click#Mode.EXPLORE
        self.prev_mode = None  # For printing purposes

        self.vendors_to_visit = []
        self.vendor_dic = {}  #list of all vendors
        self.current_vendor_index=0
        ########## PUBLISHERS ##########

        # Command pose for controller
        self.pose_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)

        #Stop remaking map after we're done exploring
        self.stop_map_update = rospy.Publisher('/stop_map_update', String, queue_size=10)

        # Command vel (used for idling)
        #self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        ########## SUBSCRIBERS ##########
        #request subcriber
        rospy.Subscriber('/delivery_request', String, self.delivery_request_callback)
        #list with objects
        rospy.Subscriber('/detector/objects', DetectedObjectList, self.objects_detected_callback)

        #find dog
        rospy.Subscriber('/detector/dog', DetectedObject, self.dog_detected_callback)

        #find way points
        rospy.Subscriber('/find_way_points', Bool, self.find_way_points_callback)


        # # Stop sign detector
        # rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)

        # High-level navigation pose
        #rospy.Subscriber('/cmd_nav', Pose2D, self.nav_pose_callback)

        # If using gazebo, we have access to perfect state
        if self.params.use_gazebo:
            rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_callback)
        self.trans_listener = tf.TransformListener()

        # If using rviz, we can subscribe to nav goal click
        # if self.params.rviz:
        if True:
            rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)
        else:
            self.x_g, self.y_g, self.theta_g = 1.5, -4., 0.
            self.mode = Mode.EXPLORE
        

    ########## SUBSCRIBER CALLBACKS ##########
    def delivery_request_callback(self,msg):
        self.stop_map_update.publish("stop")
        stores = msg.data.split(",") #[banana, apple]
        print('stores:', stores)
        for vendor in stores:
            if vendor in self.vendor_dic and vendor not in self.vendors_to_visit:
                self.vendors_to_visit.append(vendor)
        self.init_go_to_vendor()

    def dog_detected_callback(self,msg):
        print("Found the dog!")


    def objects_detected_callback(self,msg):
        """
        Received a list with the objects detected
        If if first time it sees that object, register the vendor
        """
        list_vendors_i_see=msg.objects #list of string
        #print("vendors we are seeing: ",list_vendors_i_see)
        #check if what we saw is something new
        for i in range(len(list_vendors_i_see)):
            vendor_name=list_vendors_i_see[i]
            #extract information from the message
            vendor_message=msg.ob_msgs[i]
            distance=vendor_message.distance
            if vendor_name not in ["stop_sign"]:
                if vendor_name not in self.vendor_dic:
                    #compute position in world of the vendor
                    robot_pos=(self.x, self.y , self.theta)
                    position=get_position_of_vendor(robot_pos, vendor_message)
                    #create a vendro python-object 
                    vendor= Vendor(position, vendor_name,0,distance)
                    #add vector to dictionary
                    self.vendor_dic[vendor_name] = vendor
                elif  self.vendor_dic[vendor_name].distance_detected>distance :
                    #find vendor when i was closer
                    #compute position in world of the vendor
                    robot_pos=(self.x, self.y , self.theta)
                    position=get_position_of_vendor(robot_pos, vendor_message)
                    vendor=self.vendor_dic[vendor_name]
                    vendor.position=position
                    vendor.distance_detected=distance




    
    def gazebo_callback(self, msg):
        if "turtlebot3_burger" not in msg.name:
            return

        pose = msg.pose[msg.name.index("turtlebot3_burger")]
        self.x = pose.position.x
        self.y = pose.position.y
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]
        if self.initial_pos[0]==-1:
            self.initial_pos=(self.x,self.y,self.theta)
            print("INITIAL POSITION:  ", self.initial_pos)

    def rviz_goal_callback(self, msg):
        """ callback for a pose goal sent through rviz """
        origin_frame = "/map" if self.params.mapping else "/odom"
        print("Rviz command received!")

        #save old state if different from manual or iddle
        if self.mode!=Mode.MANUAL and self.mode!=Mode.IDLE:
            self.previous_goal[0]=self.x_g
            self.previous_goal[1]=self.y_g
            self.previous_goal[2]=self.theta_g
            print("previous goal ", self.previous_goal)
            self.previous_mode=self.mode
            self.mode = Mode.MANUAL  #manual: used to take out the robot from being stuck

       
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
        
       
    # def nav_pose_callback(self, msg):
    #     self.x_g = msg.x
    #     self.y_g = msg.y
    #     self.theta_g = msg.theta
    #     self.mode = Mode.NAV

    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        dist = msg.distance

        # if close enough and in nav mode, stop
        if dist > 0 and dist < self.params.stop_min_dist and self.mode == Mode.NAV:
            self.init_stop_sign()

    def find_way_points_callback(self, msg):
        if self.mode == Mode.GO_TO_VENDOR:
            self.previous_goal = (self.x_g, self.y_g, self.theta_g)
            self.remaining_way_points = list(self.way_points)
        self.mode = Mode.WAYPOINT
        if len(self.remaining_way_points)==0:
            print("Failed at finding a path -- Tested all way points")
            return 
        new_goal = (-1, -1, -1)
        min_distance = float("inf")
        goal = np.array([self.x_g, self.y_g])
        for point in self.remaining_way_points:
            curr_point = np.array([point[0], point[1]])
            dist = np.linalg.norm(goal-curr_point)
            if dist < min_distance:
                new_goal = point
                min_distance = dist

        self.remaining_way_points.remove(new_goal)
        self.x_g, self.y_g, self.theta_g = new_goal
        self.go_to_pose((self.x_g,self.y_g,self.theta_g))
        print("send a new waypoint")



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
                print("New state: GO_TO_VENDOR")
                vendor_name=self.vendors_to_visit[self.current_vendor_index]
                print("Go to vendor ",vendor_name)
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
            print("Go to vendor: ",vendor_name)
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
        # print("desires pose"+str(point))
        # print("current pose {} {} {}".format(self.x,self.y,self.theta))
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

    def init_idle(self):
        print("New state: IDLE")
        self.stay_idle()
        self.mode=Mode.IDLE



    def stay_idle(self):
        """ sends zero velocity to stay put """
        # vel_g_msg = Twist()
        # vel_g_msg.linear.x = 0
        # vel_g_msg.linear.y = 0
        # vel_g_msg.linear.z = 0
        # vel_g_msg.angular.x = 0
        # vel_g_msg.angular.y = 0
        # vel_g_msg.angular.z = 0
        # self.cmd_vel_publisher.publish(vel_g_msg)
        pass

    def close_to(self, x, y, theta):
        """ checks if the robot is at a pose within some threshold """

        is_there=abs(x - self.x) < self.params.pos_eps and \
               abs(y - self.y) < self.params.pos_eps

        
        if theta>=0 and self.mode==Mode.EXPLORE:
            is_there=is_there and abs(theta - self.theta) < self.params.theta_eps

        return is_there

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """
        if self.mode != Mode.CROSS:
            self.previous_mode=self.mode
            self.stop_sign_start = rospy.get_rostime()
            self.mode = Mode.STOP
            

    def init_wait_on_vendor(self):
        """ initiates wait on vendor """
        self.wait_on_vendor_start = rospy.get_rostime()
        self.mode = Mode.WAIT_ON_VENDOR
        print("New state: WAIT_ON_VENDOR")

    


    def has_stopped(self):
        """ checks if stop sign maneuver is over """

        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.params.stop_time)

    def has_stopped_on_vendor(self):
        """ checks if waiting on vendor is over """

        return self.mode == Mode.WAIT_ON_VENDOR and \
               rospy.get_rostime() - self.wait_on_vendor_start > rospy.Duration.from_sec(self.params.vendor_stop_time)

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
            # print("IDLE")
            self.stay_idle()
            
        elif self.mode == Mode.POSE:
            print("POSE")
            if self.has_crossed():
                self.mode=self.previous_mode
            # else:
            #     # Moving towards a desired pose
            #     if self.close_to(self.x_g, self.y_g, self.theta_g):
            #         self.mode = Mode.IDLE
            #     else:
            #         self.go_to_pose()

        elif self.mode == Mode.STOP:
            print("STOP")
            # At a stop sign
            if self.has_stopped():
                self.init_crossing()
            else:
                self.stay_idle()

        elif self.mode == Mode.CROSS:
            print("CROSS")
            # Crossing an intersection
            self.mode=Mode.POSE

        elif self.mode == Mode.EXPLORE:
            # print("EXPLORE")
            if self.close_to(self.x_g,self.y_g,self.theta_g):
                if len(self.explore_points)>0:
                    point = self.explore_points.pop(0)
                    self.go_to_pose(point)
                else:
                    self.init_idle()
                    
            else:
                self.go_to_pose((self.x_g,self.y_g,self.theta_g))

        elif self.mode==Mode.GO_TO_VENDOR:
            # print("GO_TO_VENDOR")
            self.go_to_pose((self.x_g,self.y_g,self.theta_g))
            if self.close_to(self.x_g,self.y_g,self.theta_g):
                self.init_wait_on_vendor()

        elif self.mode == Mode.WAYPOINT:
            if self.close_to(self.x_g,self.y_g,self.theta_g):
                self.x_g=self.previous_goal[0]
                self.y_g=self.previous_goal[1]
                self.theta_g=self.previous_goal[2]
                print("going again to vendor {} {} {}".format(self.x_g,self.y_g,self.theta_g))
                self.mode=Mode.GO_TO_VENDOR
                self.go_to_pose(self.previous_goal)

        elif self.mode==Mode.WAIT_ON_VENDOR:
            self.stay_idle()
            #WAITING TIME
            if self.has_stopped_on_vendor():
                #WHEN TIME IS OVER
                if self.go_to_vendor():
                    self.mode=Mode.GO_TO_VENDOR
                else:
                    #clean list of visited vendor
                    self.vendors_to_visit=[]
                    self.mode = Mode.DELIVER 
                    print("New state: DELIVER")
        
        elif self.mode==Mode.DELIVER:
            #go to initial position 
            self.go_to_pose(self.initial_pos)
            if self.close_to(self.initial_pos[0],self.initial_pos[1],self.initial_pos[2]):
                self.init_idle()

        elif self.mode==Mode.MANUAL:
            #chick with arviz, once in reach the goal go back the state it was before
            #and the goal it was before
            if self.x_g is not None:
                self.go_to_pose((self.x_g,self.y_g,self.theta_g))
                if self.close_to(self.x_g,self.y_g,self.theta_g):
                    #if arrived, go to where it was
                    self.x_g=self.previous_goal[0]
                    self.y_g=self.previous_goal[1]
                    self.theta_g=self.previous_goal[2]
                    print("going again to previous goal goal {} {} {}".format(self.x_g,self.y_g,self.theta_g))
                    self.mode=self.previous_mode
                    self.go_to_pose(self.previous_goal)





        else:
            raise Exception("This mode is not supported: {}".format(str(self.mode)))

        ############ Code ends here ############

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        if self.mode==Mode.EXPLORE:#if we are doing autonomous exporation, retrieve firs point
            #set first point as goal
            point = self.explore_points.pop(0)
            self.go_to_pose(point)
        while not rospy.is_shutdown():
            self.loop()

            for vendor in self.vendor_dic.values():
                #print(vendor.name)
                vendor.publish_vendor_position()

            rate.sleep()



def fixAngle(a):
    if a>np.pi:
        return a-2*np.pi
    return a

#other functions
def get_position_of_vendor(robot_pos, vendor):
    x_rb = robot_pos[0]
    y_rb = robot_pos[1]
    th_rb = robot_pos[2]
    distance = vendor.distance -0.4#0.6
    th_r = fixAngle(vendor.thetaright)
    th_l = fixAngle(vendor.thetaleft)
    th_v = np.mean([th_r, th_l])
    th_out = th_rb + th_v
    x_output = x_rb+distance*np.cos(th_out)
    y_output = y_rb+distance*np.sin(th_out)
    return (x_output, y_output)

if __name__ == '__main__':

    sup = Supervisor()
    sup.run()
