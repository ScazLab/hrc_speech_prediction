#!/usr/bin/env python
import rospy
import cv2 as cv
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class DisplayPlots(object):

    def __init__(self, speech_model, speech, context, both, utter, actual=None, save_path=None):


        self.image_pub = rospy.Publisher("/robot/xdisplay", Image, queue_size=10)

        self.bridge = CvBridge()

        self.speech_model = speech_model
        self.speech = speech
        self.context = context
        self.both = both
        self.utter = utter
        self.actual = actual
        self.save_path = save_path

    def display_plots(self):

        # start with black mask
        mask = np.zeros((600, 1024, 3), np.uint8)
        # draw in white background
        cv.rectangle(mask, (0,0), (1024, 600), (255, 255, 255), -1)

        # BGR colors that will be useful
        red     = ( 44,  48, 201)
        green   = ( 60, 160,  60)
        yellow  = ( 60, 200, 200)
        blue    = (200, 162,  77)
        black   = (  0,   0,   0)

        # other helpful settings
        thickness = 3
        fontFace = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        border = 20
        max_width = 900


        title = self.utter
        textSize = cv.getTextSize(title, fontFace, fontScale, thickness)
        textOrg = ((1024 - textSize[0][0])/2, (600 + textSize[0][1])/6)
        cv.putText(mask, title, textOrg, fontFace, fontScale, black, thickness, cv.CV_AA)

        # msg = CvBridge.CvImage()
        # msg.encoding = sensor_msgs.image_encodings.BGR8
        # msg.image = mask
        # im_pub.Publish(msg.toImageMsg())
        #
        msg = self.bridge.cv2_to_imgmsg(mask, encoding="bgr8")
        try:
            print("publishing")
            # self.image_pub.publish(msg)
            rospy.sleep(1)
            self.image_pub.publish(msg)
        except CvBridgeError as e:
            print (e)

if __name__=='__main__':
    from sklearn.externals import joblib
    from hrc_speech_prediction import combined_model as cm
    rospy.init_node('display_plots')
    model_path = "/home/scazlab/ros_devel_ws/src/hrc_speech_prediction/models/"

    combined_model = joblib.load(model_path + "combined_model_0.150.15.pkl")
    # speech_model = cm.speech_model
    # vectorizer = joblib.load(model_path + "vocabulary.pkl")

    # combined_model = cm.CombinedModel(speech_model=speech_model, root=cm.Node(), vectorizer=vectorizer)
    # co?mbined_model.add_branch(["foot_2"])
    # combined_model.add_branch(["top_1"])
    # combined_model.add_branch(["foot_2", "foot_1", "leg_1"])
    # combined_model.add_branch(["chair_back", "seat", "back_1"])

    test_utter = "Pass me the blue piece with two red stripes"

    combined_model.take_action(model="both", utter=test_utter, plot=True)

    rospy.spin()
