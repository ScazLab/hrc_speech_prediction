import rospy
from std_msgs.msg import String

CONTEXT_TOPIC = '/context_history'

rospy.init_node("context_rewinder")
pub = rospy.Publisher(CONTEXT_TOPIC, String, queue_size=10)

while (not rospy.is_shutdown()):
    var = raw_input(" R \t Participant pressed the RED button accidentally. \
    \n G \t Participant pressed the GREEN button accidentally.\n\n")

    print "Publishing", var
    pub.publish(String(var))
    print
