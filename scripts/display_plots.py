#!/usr/bin/env python
import rospy
import cv2 as cv
import numpy as np
import sensor_msgs

from cv_bridge import CvBridge, CvBridgeError

class DisplayPlots(object):

    def __init__(self, speech_model, speech, context, both, utter, actual=None, save_path=None):


        self.image_pub = rospy.Publisher("/robot/xdisplay", CvBridge.CvImage, queue_size=10)

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
        cv.rectangle(mask, (0,0), (600, 1024), (255, 255, 255), -1)

        # BGR colors that will be useful
        red     = cv.Scalar( 44,  48, 201)
        green   = cv.Scalar( 60, 160,  60)
        yellow  = cv.Scalar( 60, 200, 200)
        blue    = cv.Scalar(200, 162,  77)
        black   = cv.Scalar(  0,   0,   0)

        # other helpful settings
        thickness = 3
        fontFace = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        border = 20
        max_width = 900


        title = utter
        textSize = cv.getTextSize(utter, fontFace, fontScale, thickness)
        textOrg = ((mask.cols - textSize.width)/2, (mask.rows + textSize.height)/6)
        cv.putText(mask, title, textOrg, fontFace, fontScale, black, thickness, cv.CV_AA)

        msg = CvBridge.CvImage()
        msg.encoding = sensor_msgs.image_encodings.BGR8
        msg.image = mask

        im_pub.Publish(msg.toImageMsg())
