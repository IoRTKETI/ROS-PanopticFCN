#!/usr/bin/env python

import rospkg
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def publish_image():
    # Initialize ROS node
    rospy.init_node('PanopticFCN_image', anonymous=True)

    # Create a publisher for the image topic
    rospack = rospkg.RosPack()
    path = rospack.get_path('panopticFCN')
    image_pub = rospy.Publisher('/video_source/raw', Image, queue_size=10)

    # Create a CvBridge object
    bridge = CvBridge()

    # Load the image
    image = cv2.imread(path+'/scripts/datasets/1image_rect_color18_leftImg8bit.jpg')  # Replace with the actual path to your image

    # Convert the image to the ROS format
    ros_image = bridge.cv2_to_imgmsg(image, 'bgr8')

    while not rospy.is_shutdown():
        # Publish the image
        image_pub.publish(ros_image)

if __name__ == '__main__':
    try:
        publish_image()
    except rospy.ROSInterruptException:
        pass
