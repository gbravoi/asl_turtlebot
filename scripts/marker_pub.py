#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D

#global parameters
#create pose
global x_pose
global y_pose
global theta

#initial values
x_pose = 0
y_pose =0
theta= 0

def nav_subscriber_callback(msg):
    global x_pose
    global y_pose
    global theta

    x_pose = msg.x
    y_pose = msg.y
    theta= msg.theta


def publisher():


    vis_pub = rospy.Publisher('marker_topic', Marker, queue_size=10)
    #pose_pub = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
    rospy.Subscriber('/cmd_nav', Pose2D, nav_subscriber_callback)
    rospy.init_node('marker_node', anonymous=True)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        marker = Marker()
        pose=Pose2D()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()

        # IMPORTANT: If you're creating multiple markers, 
        #            each need to have a separate marker ID.
        marker.id = 0

        marker.type = 2 # 2 sphere  10 mesh
        # marker.mesh_resource="model://dog/dog.dae"



        pose.x=x_pose
        pose.y=y_pose
        pose.theta=theta

        marker.pose.position.x = x_pose
        marker.pose.position.y = y_pose
        marker.pose.position.z = 0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        
        vis_pub.publish(marker)
        #pose_pub.publish(pose)
        print('Published marker!')
        
        rate.sleep()


if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
