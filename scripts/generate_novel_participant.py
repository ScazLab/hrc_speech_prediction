import os
import argparse

from hrc_speech_prediction import data

parser = argparse.ArgumentParser("Create two new trials from 11.BCA's data")
parser.add_argument('path', help='path to the experiment data',
                    default=os.path.curdir)

A = {
    "foot_1": ["can I have the green base with a stand with two black lines right next to the base"],
    "leg_1": ["can I have the","what are all separated out of the Metuchen each other"],
    "screwdriver_1": ["can I please have the smaller screw dropped the second smallest screwdriver"],
    "foot_2":["can I please have the big green base where the boat stripes are in the middle"],
    "leg_2": ["could I have the rod where the red stripe is on top and a blue and green striper all the way on the bottom"],
    "foot_3":["have the green stand we're both black lines at the top"],
    "leg_3": ["can I have the ride we're all three stripes at the bottom of the wooden thing"],
    "foot_4":["I had the last 3 months and then we're one black lines at the top and wants to bottom"],
    "leg_4": ["I have the rod we're all three stripes at the top"],
    "front_1": ["architrave the white corner piece for both red lines at the bottom"],
    "front_3": ["how did the school hallways in", "discovered", "can I have the white topper where both red lines are towards the opening"],
    "back_1": ["credit message","could I have the black piece with the quarter Circle we're both white lines are next to the quarter Circle"],
    "back_2": ["can I have the other black piece with two white lines where the two white lines are farther from the quarter Circle"],
    "seat": ["can I have the big wooden piece that's rectangular with the longer Stripes only blue and red stripes"],
    "leg_5": ["I'm can I have the ride with a red stripe is at the top and blue and green stripes are next to each other on the middle"],
    "top_2": ["can I have the blue corner piece"],
    "top_1": ["can I have the other blue corner piece with a red stripes are next to each other"],
    "leg_6": ["can I have The Last Ride whether red and green stripes are next to each other in a blue stripe at the bottom"],
    "chair_back": ["can I have the last wooden rectangle with us just a red stripe at the top and a blue stripe at the bottom"]}

C = {"foot_1": [],
     "leg_3": ["can I have the ride with all three lines red and green and blue at the bottom of the wooden piece"],
     "screwdriver_1": [],
     "back_1": ["can I have the black metal piece with both White Lines towards the bottom or towards a circle"],
     "leg_5": ["can I have the ride with the red line at the top and a blue and green eye in the middle"],
     "top_2": ["can I have the blue corner piece where both red lines are toward the middle they're almost next to each other"],
     "foot_2": ["can I have the Green Bay's peace we're both black lines or towards the middle"],
     "leg_4": ["can I have the ride with blue at the top"],
     "back_2": ["could I have the black piece for both White"],
     "leg_6": ["could I have the rod weather red and green lines at the top in the blue lines at the bottom"],
     "top_1": ["where are the red lines are not next to each other"],
     "leg_7": ["could I have the rod where the red and green lines in the middle and Plies bottom"],
     "chair_back": ["smaller wooden rectangular", "whether it is a blue line at the top and a red light at the bottom"],
     "foot_3": ["can I have the", "can I have the Green Bay's PS4 both black lines at the top"],
     "leg_1": ["how can I have the rod where the red light at the top the green line in the middle and a blue eyes at the bottom"],
     "front_1": ["can I have the white corner piece for both the red lines are towards the bottom"],
     "foot_4": ["vodka the Green Bay's peace"],
     "leg_2": ["can I have The Rock where one red blue"],
     "front_3": ["can I have the white PS4 about my lines at the top"]}


B = {"foot_1": ["give me the green cylinder with","two","at the bottom near the base"],
     "foot_2": ["Now give me the green cylinder with the two stripes in the middle"],
     "foot_3": ["now I need the green cylinder with the two stripes at the top"],
     "foot_4": ["give me the green cylinder that's left"],
     "leg_1": ["give me the wooden plank the one with the weather three stripes are separated"],
     "screwdriver_1": ["give me the screwdriver with two blue stripes near the metal shaft"],
     "leg_2": ["now I need the wooden part where the red stripe", "is it at the top but it's separated from the other ones"],
     "leg_3": ["now hand me the Wooden Park where the three stripes are at the bottom"],
     "leg_4": ["now the other one with the street with the three stripes together at the top"],
     "front_1": ["now I need the white cylinder with a base where the two red stripes are near the base"],
     "front_3": ["get me another white cylinder now with the two stripes near the top"],
     "back_1": ["now imma need a black cylinder with two white stripes"],
     "back_2": ["Now give me the remaining black cylinder with white stripes"],
     "seat": ["give me the Wooden Park where the red stripe is on the left"],
     "leg_5": ["give me the wooden part with the symbol that looks like an H"],
     "leg_6": ["now the other one with a symbol that looks like an a"],
     "top_2": ["triangle shaped","peace where the two pairs of red striped are close together"],
     "top_1": ["give me the other triangle shaped blue piece"],
     "leg_7": ["give me the Woodland Park","three colored stripes"],
     "chair_back": ["give me the remaining Wooden Park"]}


d_actions  = ["leg_7", "top_2", "top_1", "chair_back", "screwdriver_1",
              "foot_3", "leg_3", "foot_4", "leg_4", "foot_1", "leg_1",
              "foot_2", "leg_2", "front_1", "front_3", "back_1", "back_2",
              "seat", "leg_5", "leg_6"]

e_actions = ["screwdriver_1", "foot_2", "leg_2", "foot_4", "leg_4", "foot_3",
             "leg_3", "foot_1", "leg_1", "back_1", "back_2", "front_1",
             "front_3", "seat", "leg_5", "leg_6", "leg_7", "top_2",
             "top_1", "chair_back"]

def get_pairs(actions):
    t_actions = [(a, 0, 0) for a in actions]
    t_utterances = []

    for a in actions:
        t_u = []
        for u in B[a]:
            t_u.append((u, 0, 0))
        t_utterances.append(t_u)

    return zip(t_actions ,t_utterances)

trials = [data.Trial('D', get_pairs(d_actions), 0),
          data.Trial('E', get_pairs(e_actions), 0)]

new_data = data.TrainData({'DE': data.Session(trials)})
args = parser.parse_args()
new_data.dump("/home/ros/ros_ws/src/hrc_speech_prediction/new_participant.json")
