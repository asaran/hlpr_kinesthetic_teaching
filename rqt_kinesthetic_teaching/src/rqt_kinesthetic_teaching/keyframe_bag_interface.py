"""
keyframe_bag_parser.py

Parses and reads .bag files containing keyframe recordings generated by demonstration.py

"""

import actionlib
import os
import rosbag
import roslib
import rospy
import time

from hlpr_record_demonstration.playback_demonstration_action_server import PlaybackKFDemoAction
from hlpr_record_demonstration.msg import PlaybackKeyframeDemoAction, PlaybackKeyframeDemoGoal, PlaybackKeyframeDemoResult, PlaybackKeyframeDemoFeedback
from std_msgs.msg import Int32, String

class ParseException(Exception):
    pass

class KeyframeBagInterface():
    
    def __init__(self):
        self.bag = None
        self.client = None

    def parse(self, file):
        """
        Parses the keyframes from a bag file
        Adapted from do_playback_keyframe_demo() in playback_demonstration_action_server.py
        """
        if not os.path.isfile(file):
            raise ParseException("File does not exist")

        with rosbag.Bag(file) as bag:
            self.bag = bag
            parsed = []

            all_topics = self.bag.get_type_and_topic_info().topics.keys()
            GRIPPER_TOPIC = "gripper/stat"
            gripper_topics = [x for x in all_topics if GRIPPER_TOPIC in x]

            for topic, msg, time in self.bag.read_messages():
                # Grabber and joint messages will be linked if they are within 100 ms of each other
                # More precision runs into problems described at https://stackoverflow.com/a/22155830
                time_key = round(time.to_time(), 1)

                data = [item for item in parsed if item["time"] == time_key]
                shouldAppend = False
                if len(data) > 0:
                    data = data[0]
                else:
                    data = dict(
                        time = time_key,
                        data = dict()
                    )    
                    shouldAppend = True

                if topic in gripper_topics:
                    # Gripper has different data
                    data["data"][topic] = dict(
                        position = msg.position,
                        requested_position = msg.requested_position
                    )
                elif topic == "joint_states":
                    for i, name in enumerate(msg.name):
                        data["data"]["/{}/{}".format(topic, name)] = dict(
                            position = msg.position[i],
                            velocity = msg.velocity[i],
                            effort = msg.effort[i]
                        )
                elif topic == "eef_pose":
                    data["data"]["/{}".format(topic)] = dict(
                        position_x = msg.position.x,
                        position_y = msg.position.y,
                        position_z = msg.position.z,
                        orientation_x = msg.orientation.x,
                        orientation_y = msg.orientation.y,
                        orientation_z = msg.orientation.z,
                        orientation_w = msg.orientation.w
                    )
                elif topic == "object_location":
                    for item in msg.objects:
                        data["data"]["/{}/{}".format(topic, item.label)] = dict(
                            label = item.label,
                            probability = item.probability,
                            position_x = item.pose.position.x,
                            position_y = item.pose.position.y,
                            position_z = item.pose.position.z,
                            orientation_x = item.pose.orientation.x,
                            orientation_y = item.pose.orientation.y,
                            orientation_z = item.pose.orientation.z,
                            orientation_w = item.pose.orientation.w
                        )
                else:
                    rospy.loginfo("Found unknown topic while parsing: {}".format(topic))

                if shouldAppend:
                    parsed.append(data)
            return parsed

    def playInit(self):
        self.client = actionlib.SimpleActionClient("playback_keyframe_demo", PlaybackKeyframeDemoAction)
        self.client.wait_for_server()

    def play(self, file, cb):
        """
        Plays the keyframes from a bag file
        Calls do_playback_keyframe_demo() in playback_demonstration_action_server.py
        """
        if not self.client:
            raise ParseException("Play not initialized")
        if not os.path.isfile(file):
            raise ParseException("File does not exist")

        time.sleep(1.0)

        goal = PlaybackKeyframeDemoGoal()
        goal.bag_file_name = file
        goal.eef_only = False
        goal.target_topic = "joint_states"

        self.client.send_goal(goal, feedback_cb=cb)