var width  = 640,
    height = 480;

var obj_dict = {
    200: "seat",
    201: "chair back",
    150: "leg 1",
    151: "leg 2",
    152: "leg 3",
    153: "leg 4",
    154: "leg 5",
    155: "leg 6",
    156: "leg 7",
    10:  "foot 1",
    11:  "foot 2",
    12:  "foot 3",
    13:  "foot 4",
    14:  "front 1",
    15:  "front 2",
    16:  "top 1",
    17:  "top 2",
    18:  "back 1",
    19:  "back 2",
    20:  "screwdriver 1",
    21:  "screwdriver 2",
    22:  "front 3",
    23:  "front 4"
};

var EVENT_STARTED = 0;
var EVENT_STOPPED = 1;
var EVENT_DECODED = 2;
var EVENT_FAILED  = 3;

// Create a left and right SVG, corresponding to the left and right tables
// that baxter picks items up from
var left_svg = d3.select("#left-svg-container").select("svg")
//responsive SVG needs these 2 attributes and no width and height attr
    .attr('preserveAspectRatio', 'xMinYMin meet')
    .attr('viewBox', '0 0 ' + width + ' ' + height)
//class to make it responsive
    .classed('svg-content-responsive', true);

var right_svg = d3.select("#right-svg-container").select("svg")
//responsive SVG needs these 2 attributes and no width and height attr
    .attr('preserveAspectRatio', 'xMinYMin meet')
    .attr('viewBox', '0 0 ' + width + ' ' + height)
//class to make it responsive
    .classed('svg-content-responsive', true);


// Function for drawing polygon from corner coordinates
var lineFunction = d3.svg.line()
    .x(function(d) {
        return d.x; })
    .y(function(d) { return d.y; })
    .interpolate("linear");

// Visualizes aruco data from left arm
leftAruco.subscribe(function(msg){

    var left_markers = msg.markers;
    arucoCallback(left_markers,left_svg);
});


rightAruco.subscribe(function(msg){

    var  right_markers = msg.markers;
    arucoCallback(right_markers,right_svg);
});


leftArmInfo.subscribe(function(msg){
    armInfoCallback(msg, "left");
});

rightArmInfo.subscribe(function(msg){
    armInfoCallback(msg, "right");
});

speech2Text.subscribe(function(msg){
    speech2TextCallback(msg);
});

function armInfoCallback(msg,arm){

    console.log("Receiving arm info!: " + msg.state);
    var s = d3.select("#" + arm + "_baxter_state");
    var a = d3.select("#" + arm + "_baxter_action");
    var o = d3.select("#" + arm + "_baxter_object");

    s.select("text")
        .text("STATE: " + msg.state);

    a.select("text")
        .text("ACTION: " + msg.action);

    o.select("text")
        .text("OBJECT: " + msg.object);
}

function speech2TextCallback(msg){

    var s = d3.select("#speech2TextStatus");
    var t = d3.select("#speech2TextTranscription");

    if (msg.event == EVENT_STARTED) {
        s.classed("bg-info", true)
            .text("[" + msg.utterance_id + "]");
    } else if (msg.event == EVENT_STOPPED) {
        s.classed("bg-info", false)
            .text("");
    } else if (msg.event == EVENT_DECODED) {
        t.classed("alert-danger", false)
         .classed("alert-info", true)
            .text("[" + msg.utterance_id + "] " + msg.transcript.transcript);
    } else if (msg.event == EVENT_FAILED) {
        t.classed("alert-info", false)
         .classed("alert-danger", true)
            .text("[" + msg.utterance_id + "]  Failed to decode speech");
    } else {
        t.classed("alert-info", false)
         .classed("alert-danger", true)
            .text("Unkown event: " + msg.event);
    }
}

function arucoCallback(markers,s){

    s.selectAll("*").remove();

    var right_objs = s.selectAll("g")
        .data(markers).enter().append("g");

    // UPDATES existing objects-------------------
    right_objs
        .append("text")
        .text(function(d){
            return obj_dict[d.id] === undefined? "Unknown " +d.id : obj_dict[d.id];
        })
        .attr("x", function(d){
            //console.log("X cord of item is :" + d.center.x);
            return d.center.x;})
        .attr("y", function(d){
            //console.log("y cord of item is :" + d.center.y);
            return d.center.y;})
        .attr("text-anchor", "middle");


    for(var d in markers){
        right_objs.append("path")
            .attr("d", lineFunction( markers[d].corners) + " z")
            .attr("stroke", "blue")
            .attr("stroke-width", 2)
            .attr("fill", "none");
    }
    return;
}
