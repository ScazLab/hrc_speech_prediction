import rospy
from std_msgs.msg import String

CONTEXT_TOPIC = '/context_history'

rospy.init("context_rewinder")
pub = rospy.Publisher(CONTEXT_TOPIC, String, queue_size=10)

while (True):
    var = raw_input("R \t Participant pressed the red button accidentally. \
    \nG \t Participant pressed the green button accidentally.")

    print "Publishing", var
    pub.publish(String(var))
    print
