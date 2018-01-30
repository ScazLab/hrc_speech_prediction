#!/usr/bin/env python
import rospy
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import PIL
from cStringIO import StringIO

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

        plt.ioff()
        X = np.arange(len(self.both))
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        fig = plt.figure(figsize=(12.84,5.81))
        ax = fig.add_subplot(1, 1, 1)

        # Want to normalize 'both' probs for easier visual comparison
        nrmlz = 1.0 / sum(self.both)

        ax.bar(X - 0.2, self.speech, width=0.2, color='r', align='center')
        ax.bar(X, self.context, width=0.2, color='b', align='center')
        ax.bar(X + 0.2, self.both * nrmlz, width=0.2, color='g', align='center')

        ax.legend(('Speech', 'Context', 'Both'))

        rects = ax.patches
        max_prob = max(self.both * nrmlz)

        # This draws a star above most probable action
        for r in rects:
            if r.get_height() == max_prob:
                ax.text(
                    r.get_x() + r.get_width() / 2,
                    r.get_height() * 1.01,
                    '*',
                    ha='center',
                    va='bottom')

        if self.actual:
            ax.text(self.speech_model.actions.index(self.actual), max_prob, "$")

        plt.xticks(X, self.speech_model.actions, rotation=60)
        plt.title(self.utter)

        buffer_ = StringIO()
        plt.savefig(buffer_, format="png", bbox_inches='tight', pad_inches=.1)

        buffer_.seek(0)

        image = PIL.Image.open(buffer_)
        ar = np.asarray(image)
        colored = cv.cvtColor(ar, cv.COLOR_RGB2BGR)
        # convert and publish image
        msg = self.bridge.cv2_to_imgmsg(colored, "bgr8")
        try:
            rospy.sleep(1)
            self.image_pub.publish(msg)
        except CvBridgeError as e:
            print (e)

        if self.save_path:
            plt.savefig(self.save_path)
        else:
            plt.show(block=False)

    def rotate(self, src, pt, angle):
        rows = 600
        cols = 1024
        M = cv.getRotationMatrix2D(pt,angle,1)
        dst = cv.warpAffine(src, M, (cols, rows))
        return dst


if __name__=='__main__':
    from sklearn.externals import joblib
    from hrc_speech_prediction import combined_model as cm
    rospy.init_node('display_plots')
    model_path = "/home/scazlab/ros_devel_ws/src/hrc_speech_prediction/models/"

    combined_model = joblib.load(model_path + "combined_model_0.150.15.pkl")

    test_utter = "Pass me the blue piece with two red stripes"

    combined_model.take_action(model="both", utter=test_utter, plot=True)

    rospy.spin()
