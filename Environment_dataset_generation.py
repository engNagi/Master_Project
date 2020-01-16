import random
import numpy as np
import pandas as pd
import ai2thor.controller
from ai2thor.controller import Controller, BFSController

random.seed(123)
np.random.seed(123)


def noop(self):
    pass


ai2thor.controller.Controller.lock_release = noop
ai2thor.controller.Controller.unlock_release = noop
ai2thor.controller.Controller.prune_releases = noop


#    This axis has “right hand” facing with respect to the forward Z-Axis,
#    Y-axis pointing upward, z-axis pointing forward, x axis  pointing to the left
class Environment(object):
    def __init__(self,
                 fov=60.0,
                 action_n=6,
                 camera_Y=0.675,
                 grid_size=0.20,
                 visibility_distance=1.5,
                 player_screen_width=300,
                 player_screen_height=300,
                 full_scrn=False,
                 depth_image=False,
                 class_image=False,
                 top_view_cam=False,
                 object_image=False,
                 third_party_cam=False,
                 random_init=False,
                 random_goals=False,
                 scene="FloorPlan220",
                 object_name="Television"):

        # self.scene = scene
        self.grid_size = grid_size
        self.depth_image = depth_image
        self.class_image = class_image
        self.object_image = object_image
        self.visibility_distance = visibility_distance
        self.camera_Y = camera_Y
        self.fov = fov
        self.scene = scene
        self.object_name = object_name
        self.player_screen_width = player_screen_width
        self.player_screen_height = player_screen_height
        self.top_view_cam = top_view_cam
        self.third_party_cam = third_party_cam
        self.full_scrn = full_scrn
        self.random_init = random_init
        self.orientations = [0.0, 90.0, 180.0, 270.0, 360.0]
        self.action_n = action_n
        self.random_goal = random_goals

        self.ctrl = Controller()  # headless=True

    def make(self):
        self.ctrl.start()   # x_display="50"
        self.ctrl.reset(self.scene)

    def reset(self):
        random_goal_position = 0
        self.ctrl.reset(self.scene)

        if self.top_view_cam:
            self.ctrl.step(dict(action="ToggleMapView"))

        self.ctrl.step(dict(action="Initialize",
                            gridSize=self.grid_size,
                            renderDepthImage=self.depth_image,
                            renderClassImage=self.class_image,
                            renderObjectImage=self.object_image,
                            visibilityDistance=self.visibility_distance,
                            cameraY=self.camera_Y,
                            fieldOfView=self.fov))

        if self.random_init:
            agent_random_spwan = self.agent_random_init()

        if self.random_goal:
            random_goal_position = self.random_goal_position()

        agent_position, agent_rotation, agent_pose = self.agent_properties()

        #   object position, visibility nad distance from agent to specified object
        obj_position, obj_visibility, obj_agent_distance = self.object_properties()
        #goal = list(obj_position.values())
        try:
            np.array_equal(np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values())), agent_position)
        except:
            print("agent init position does not equal to agent position attribute")
        first_person_obs = self.ctrl.last_event.frame
        agent_pos_dis = np.linalg.norm(random_goal_position - agent_position)

        return first_person_obs, agent_position, random_goal_position, agent_pos_dis, agent_pose, self.object_name

    def step(self, action, goal, distance):
        #   move right
        if action == 0:
            self.ctrl.step(dict(action="MoveRight"))
            reward, done, distance_, first_person_obs, collision, agent_position = self.post_action_state(goal, distance)
        #   right rotate
        elif action == 1:
            self.ctrl.step(dict(action='RotateRight'))
            reward, done, distance_, first_person_obs, collision, agent_position = self.post_action_state(goal, distance)
        #   left rotate
        elif action == 2:
            self.ctrl.step(dict(action="RotateLeft"))
            reward, done, distance_, first_person_obs, collision, agent_position = self.post_action_state(goal, distance)
        #   move left
        elif action == 3:
            self.ctrl.step(dict(action='MoveLeft'))
            reward, done, distance_, first_person_obs, collision, agent_position = self.post_action_state(goal, distance)
        #   move Ahead
        elif action == 4:
            self.ctrl.step(dict(action="MoveAhead"))
            reward, done, distance_, first_person_obs, collision, agent_position = self.post_action_state(goal, distance)
        #   Move back
        elif action == 5:
            self.ctrl.step(dict(action="MoveBack"))
            reward, done, distance_, first_person_obs, collision, agent_position = self.post_action_state(goal, distance)
        #   Crouch
        elif action == 6:
            self.ctrl.step(dict(action="Crouch"))
            reward, done, distance_, first_person_obs, collision, agent_position = self.post_action_state(goal, distance)
        #   Stand
        elif action == 7:
            self.ctrl.step(dict(action="Stand"))
            reward, done, distance_, first_person_obs, collision, agent_position = self.post_action_state(goal, distance)
        #   Look up
        elif action == 8:
            self.ctrl.step(dict(action="LookUp"))
            reward, done, distance_, first_person_obs, collision, agent_position = self.post_action_state(goal, distance)
        #   Look down
        elif action == 9:
            self.ctrl.step(dict(action="LookDown"))
            reward, done, distance_, first_person_obs, collision, agent_position = self.post_action_state(goal, distance)

        return first_person_obs, agent_position, distance_, done, reward, collision

    def agent_properties(self):
        agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
        agent_rotation = np.array(list(self.ctrl.last_event.metadata["agent"]["rotation"].values()))
        agent_pose = np.concatenate((agent_position, agent_rotation), axis=0)

        return agent_position, agent_rotation, agent_pose

    def get_reachable_position(self):
        self.ctrl.step(dict(action='GetReachablePositions'))
        return pd.DataFrame(self.ctrl.last_event.metadata["reachablePositions"]).values

    def action_sampler(self):
        return np.random.choice(self.action_n)

    def object_properties(self):
        self.obj = self.ctrl.last_event.metadata["objects"]
        for obj in self.obj:
            if self.object_name in obj["name"]:
                goal_object = obj
                break
        obj_position = goal_object["position"]
        obj_visibility = goal_object["visible"]
        obj_agent_distance = goal_object["distance"]
        return obj_position, obj_visibility, obj_agent_distance

    def agent_random_init(self):
        reachable_positions = self.get_reachable_position()
        idx = np.random.choice(len(reachable_positions))
        angle = np.random.choice(self.orientations)
        x_pos = reachable_positions[idx][0]
        y_pos = reachable_positions[idx][1]
        z_pos = reachable_positions[idx][2]
        agent_pose = self.ctrl.step(dict(action="Teleport", x=x_pos, y=y_pos,
                                         z=z_pos))

        return agent_pose

    def random_goal_position(self):
        positions = self.get_reachable_position()
        idx = np.random.choice(len(positions))
        position = positions[idx]

        return position

    def post_action_state(self, goal, dist):
        # _, visible, obj_agent_dis_ = self.object_properties()
        agent_position, agent_rotation, agent_pose = self.agent_properties()
        first_person_obs = self.ctrl.last_event.frame
        dist_ = np.linalg.norm(goal - agent_position)
        collide = not self.ctrl.last_event.metadata["lastActionSuccess"]
        if dist_ == 0:
            reward = 0
            done = True
        elif collide:
            reward = -1
            done = True
        elif dist_ < dist:
            reward = -1 + (dist - dist_)
            done = False
        else:
            reward = -1
            done = False

        return reward, done, dist_, first_person_obs, collide, agent_position

        return reward, done, visible, obj_agent_dis_, first_person_obs, collide

