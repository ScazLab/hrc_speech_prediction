#!/usr/bin/env python
import rospy
import cv2

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class DisplayPlots(object):

    def __init__(self):
        self.image_pub = rospy.Publisher("/robot/xdisplay", Image, queue_size=10)

        self.bridge = CvBridge()
        self.prob_sub = rospy.Subscriber("/hrc_speech_pred/probabilities", String, self.display_plots)

    def display_plots(self, data):
        # make cv image
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

def main():
    dp = DisplayPlots()
    rospy.init_node('display_plots', anonymous=True)
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
