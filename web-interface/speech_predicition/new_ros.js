// Connecting to ROS
// -----------------
var beginning = Date.now();
var ros = new ROSLIB.Ros();

// If there is an error on the backend, an 'error' emit will be emitted.
ros.on('error', function(error) {
  document.getElementById('connecting').style.display = 'none';
  document.getElementById('connected').style.display = 'none';
  document.getElementById('closed').style.display = 'none';
  document.getElementById('error').style.display = 'block';
  document.getElementById('troubleshooting').style.display = 'inline-block';
  console.log(error);
});

// Find out exactly when we made a connection.
ros.on('connection', function() {
  console.log('Connection made!');
  document.getElementById('connecting').style.display = 'none';
  document.getElementById('error').style.display = 'none';
  document.getElementById('closed').style.display = 'none';
  document.getElementById('connected').style.display = 'block';
});

ros.on('close', function() {
  console.log('Connection closed.');
  document.getElementById('connecting').style.display = 'none';
  document.getElementById('connected').style.display = 'none';
  document.getElementById('closed').style.display = 'inline-block';
  document.getElementById('error').style.display = 'inline-block';
});

// Guess connection of the rosbridge websocket
function getRosBridgeHost() {
  if (window.location.protocol == 'file:') {
    return '192.168.1.3';
  } else {
    return window.location.hostname;
  }
}

var rosBridgePort = 9090;
// Create a connection to the rosbridge WebSocket server.
ros.connect('ws://' + getRosBridgeHost() + ':' + rosBridgePort);

// First, we create a Topic object with details of the topic's name and message type.
var logTopic = new ROSLIB.Topic({
  ros : ros,
  name : '/web_interface/log',
  messageType : 'std_msgs/String'
});

// First, we create a Topic object with details of the topic's name and message type.
var elemPressed = new ROSLIB.Topic({
  ros : ros,
  name : '/web_interface',
  messageType : 'std_msgs/String'
});

// Topic for error passing to the left arm
var errorPressedL = new ROSLIB.Topic({
  ros : ros,
  name : '/robot/digital_io/left_lower_button/state',
  messageType : 'baxter_core_msgs/DigitalIOState'
});

// Topic for error passing to the right arm
var errorPressedR = new ROSLIB.Topic({
  ros : ros,
  name : '/robot/digital_io/right_lower_button/state',
  messageType : 'baxter_core_msgs/DigitalIOState'
});

var leftAruco  = new ROSLIB.Topic({
    ros : ros,
    name: '/baxter_aruco_left/markers',
    messageType : 'aruco_msgs/MarkerArray'
});

var rightAruco  = new ROSLIB.Topic({
    ros : ros,
    name: '/baxter_aruco_right/markers',
    messageType : 'aruco_msgs/MarkerArray'
});

var rightArmInfo = new ROSLIB.Topic({
    ros : ros,
    name: '/action_provider/right/state',
    messageType : 'human_robot_collaboration_msgs/ArmState'
});

var leftArmInfo = new ROSLIB.Topic({
    ros : ros,
    name: '/action_provider/left/state',
    messageType : 'human_robot_collaboration_msgs/ArmState'
});

var speech2Text = new ROSLIB.Topic({
    ros : ros,
    name: '/speech_to_text/log',
    messageType : 'ros_speech2text/event'
});

// Service Client to interface with the left arm
var leftArmService  = new ROSLIB.Service({
  ros : ros,
  name: '/action_provider/service_left',
  messageType : 'human_robot_collaboration_msgs/DoAction'
});

// Service Client to interface with the right arm
var rightArmService = new ROSLIB.Service({
  ros : ros,
  name: '/action_provider/service_right',
  messageType : 'human_robot_collaboration_msgs/DoAction'
});

var startExperiment = new ROSLIB.Service({
    ros : ros,
    name: '/rosbag/start',
    messageType :'std_srvs/Empty'
});

var stopExperiment = new ROSLIB.Service({
    ros : ros,
    name: '/rosbag/stop',
    messageType :'std_srvs/Empty'
});

// Add a callback for any element on the page
function callback(e) {
    var e = window.e || e;

    if (e.target.tagName == 'BUTTON')
    {
        var obj = e.target.firstChild.nodeValue;
        var message = new ROSLIB.Message({
            data: obj
        });

        elemPressed.publish(message);
    }
    return;
}

if (document.addEventListener)
    document.addEventListener('click', callback, false);
else
    document.attachEvent('onclick', callback);
