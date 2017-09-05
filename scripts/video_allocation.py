#!/usr/bin/env python

import re

import rospy
from human_robot_collaboration.controller import BaseController


class AllocationController(BaseController):

    OBJECTS_LEFT = "action_provider/objects_left"
    OBJECTS_RIGHT = "action_provider/objects_right"
    SCENARIO = 0

    @staticmethod
    def _parse_objects(obj_dict):
        obj_parser = re.compile('(.*)_[0-9]+$')
        d = {}
        for o in obj_dict:
            m = obj_parser.match(o)
            if m is None:
                new_o = o
            else:
                new_o = m.group(1)
            if new_o not in d:
                d[new_o] = []
            d[new_o].append(obj_dict[o])
        return d

    def __init__(self, *args, **kargs):
        super(AllocationController, self).__init__(*args, **kargs)
        self.objects_left = self._parse_objects(
            rospy.get_param(self.OBJECTS_LEFT))
        self.objects_right = self._parse_objects(
        rospy.get_param(self.OBJECTS_RIGHT))
        print('Found {} on left and {} objects on right arm.'
              ''.format(list(self.objects_left),
                        list(self.objects_right)))
        self.screwdriver = self.objects_right['screwdriver']
        self.backets = self.objects_right['brackets_box']
        self.screws = self.objects_right['screws_box']
        self.top = self.objects_left['table_top']
        self.leg = self.objects_left['leg']

    def _run(self):
        if self.SCENARIO < 2:
            self.say("I'm going to bring the screw box")
            self.action_right('get_pass', self.screws, wait=True)
            if self.SCENARIO == 0:
                self.say("I'm going to bring the top. "
                        "Can you please take the screwdriver?")
                self.action_left('get_pass', self.top, wait=True)
            elif self.SCENARIO == 1:
                self.say("I'm going to bring the screwdriver. "
                         "Can you please take the table top?.")
                self.action_right('get_pass', self.screwdriver, wait=True)
        else:
            self.action_right('get_pass', self.screws, wait=False)
            rospy.sleep(6)
            self.action_left('get_pass', self.leg, wait=False)
            self.action_right('get_pass', self.screwdriver, wait=False)
            if self.SCENARIO < 4:
                self.action_right('hold_leg', [60, 180], wait=True)
            if self.SCENARIO == 2:
                self.action_left('get_pass', self.leg, wait=True)
                self.action_right('hold_leg', [60, 180], wait=True)
            elif self.SCENARIO == 3:
                self.action_left('get_pass', self.top, wait=True)
                self.action_right('hold_top', [60, 180], wait=True)


c = AllocationController()
c.SCENARIO = 1
c.run()
