# TODO
#  genalize the goal such that returns the position where
#  the corresponding object would be visible to the agent
#   Creating dataset for the AVE
#   Train The AVE with the gathered dataset


from ai2thor.controller import Controller, BFSController
import numpy as np
import pandas as pd
import ai2thor.controller
import random


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
                 camera_Y=0.675,
                 grid_size=0.15,
                 visibility_distance=1.5,
                 player_screen_width=300,
                 player_screen_height=300,
                 action_n=6,
                 full_scrn=False,
                 depth_image=False,
                 class_image=False,
                 top_view_cam=False,
                 object_image=False,
                 third_party_cam=False,
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
        self.object_name = object_name
        self.player_screen_width = player_screen_width
        self.player_screen_height = player_screen_height
        self.top_view_cam = top_view_cam
        self.third_party_cam = third_party_cam
        self.full_scrn = full_scrn
        self.orientations = [0.0, 90.0, 180.0, 270.0, 360.0]
        self.action_n = action_n
        self.scene = scene

        self.ctrl = Controller()

    def make(self):
        self.ctrl.start()
        self.ctrl.reset(self.scene)

    def reset(self, random_init=False):
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
        if random_init:
            reachable_positions = self.get_reachable_position()
            idx = np.random.choice(len(reachable_positions))
            angle = np.random.choice(self.orientations)
            x_pos = reachable_positions[idx][0]
            y_pos = reachable_positions[idx][1]
            z_pos = reachable_positions[idx][2]
            self.ctrl.step(dict(action="Teleport", x=x_pos, y=y_pos,
                                z=z_pos))

        agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
        agent_rotation = np.array(list(self.ctrl.last_event.metadata["agent"]["rotation"].values()))
        agent_pose = np.concatenate((agent_position, agent_rotation), axis=0)

        self.obj = self.ctrl.last_event.metadata["objects"]
        obj = random.sample(self.obj, 1)[0]

        for obj in self.obj:
            if self.object_name in obj["name"]:
                obj_ = obj
        self.obj_pos = obj["position"]
        self.visible = obj["visible"]
        self.obj_agent_dis = obj["distance"]

        first_person_obs = self.ctrl.last_event.frame

        self.goal = random.choice(self.get_reachable_position())
        return first_person_obs, agent_position, agent_pose, self.object_name, self.goal, self.obj_agent_dis

    def take_action(self, action):
        #   move right
        if action == 0:
            self.ctrl.step(dict(action="MoveRight"))
            if self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = 0
                if self.visible:
                    reward = 1
            else:
                reward = -1
        #   right rotate
        elif action == 1:
            event = self.ctrl.step(dict(action='RotateRight'))
            if self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = 0
                if self.visible:
                    reward = 1
            else:
                reward = -1
        #   left rotate
        elif action == 2:
            self.ctrl.step(dict(action="RotateLeft"))
            if self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = 0
                if self.visible:
                    reward = 1
            else:
                reward = -1
        #   move left
        elif action == 3:
            self.ctrl.step(dict(action='MoveLeft'))
            if self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = 0
                if self.visible:
                    reward = 1
            else:
                reward = -1
        #   move Ahead
        elif action == 4:
            self.ctrl.step(dict(action="MoveAhead"))
            if self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = 0
                if self.visible:
                    reward = 1
            else:
                reward = -1
        #   Move back
        elif action == 5:  # move_back action
            self.ctrl.step(dict(action="MoveBack"))
            if self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = 0
                if self.visible:
                    reward = 1
            else:
                reward = -1
        elif action == 6:  # crouch action
            self.ctrl.step(dict(action="Crouch"))
            if self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = 0
                if self.visible:
                    reward = 1
            else:
                reward = -1
        elif action == 7:  # stand action
            self.ctrl.step(dict(action="Stand"))
            if self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = 0
                if self.visible:
                    reward = 1
            else:
                reward = -1
        elif action == 8:  # lookup action
            self.ctrl.step(dict(action="LookUp"))
            if self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = 0
                if self.visible:
                    reward = 1
            else:
                reward = -1
        elif action == 9:  # lookdown action
            self.ctrl.step(dict(action="LookDown"))
            if self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = 0
                if self.visible:
                    reward = 1
            else:
                reward = -1
        #   1st person camera
        first_person_obs = self.ctrl.last_event.frame
        #    Third party_cam "From top"
        third_cam_obs = self.ctrl.last_event.third_party_camera_frames
        # third_cam_obs = np.squeeze(third_cam_obs, axis=0)
        # done condition when the last action was successful inverted
        done = not self.ctrl.last_event.metadata["lastActionSuccess"]
        #   agent position
        agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
        agent_rotation = np.array(list(self.ctrl.last_event.metadata["agent"]["rotation"].values()))
        agent_pose = np.concatenate((agent_position, agent_rotation), axis=0)

        return first_person_obs, agent_position, agent_pose, done, reward

    def modified_take_action(self, action):
        #   move right
        if action == 0:
            self.ctrl.step(dict(action="MoveRight"))
            agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
            if not self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = -1
                done = True
            elif np.array_equal(self.goal, agent_position):
                reward = 10
                done = True
            else:
                reward = -1
                done = False
        #   right rotate
        elif action == 1:
            self.ctrl.step(dict(action='RotateRight'))
            agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
            if not self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = -1
                done = True
            elif np.array_equal(self.goal, agent_position):
                reward = 10
                done = True
            else:
                reward = -1
                done = False
        #   left rotate
        elif action == 2:
            self.ctrl.step(dict(action="RotateLeft"))
            agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
            if not self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = -1
                done = True
            elif np.array_equal(self.goal, agent_position):
                reward = 10
                done = True
            else:
                reward = -1
                done = False
        #   move left
        elif action == 3:
            self.ctrl.step(dict(action='MoveLeft'))
            agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
            if not self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = -1
                done = True
            elif np.array_equal(self.goal, agent_position):
                reward = 10
                done = True
            else:
                reward = -1
                done = False
        #   move Ahead
        elif action == 4:
            self.ctrl.step(dict(action="MoveAhead"))
            agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
            if not self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = -1
                done = True
            elif np.array_equal(self.goal, agent_position):
                reward = 10
                done = True
            else:
                reward = -1
                done = False
        #   Move back
        elif action == 5:
            self.ctrl.step(dict(action="MoveBack"))
            agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
            if not self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = -1
                done = True
            elif np.array_equal(self.goal, agent_position):
                reward = 10
                done = True
            else:
                reward = -1
                done = False
        # crouch action
        elif action == 6:
            self.ctrl.step(dict(action="Crouch"))
            agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
            if not self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = -1
                done = True
            elif np.array_equal(self.goal, agent_position):
                reward = 10
                done = True
            else:
                reward = -1
                done = False
        # stand action
        elif action == 7:
            self.ctrl.step(dict(action="Stand"))
            agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
            if not self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = -1
                done = True
            elif np.array_equal(self.goal, agent_position):
                reward = 10
                done = True
            else:
                reward = -1
                done = False
        # lookup action
        elif action == 8:
            self.ctrl.step(dict(action="LookUp"))
            agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
            if not self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = -1
                done = True
            elif np.array_equal(self.goal, agent_position):
                reward = 10
                done = True
            else:
                reward = -1
                done = False
        # lookdown action
        elif action == 9:
            self.ctrl.step(dict(action="LookDown"))
            agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
            if not self.ctrl.last_event.metadata["lastActionSuccess"]:
                reward = -1
                done = True
            elif np.array_equal(self.goal, agent_position):
                reward = 10
                done = True
            else:
                reward = -1
                done = False
        #   1st person camera
        #    Third party_cam "From top"
        # third_cam_obs = self.ctrl.last_event.third_party_camera_frames
        # third_cam_obs = np.squeeze(third_cam_obs, axis=0)
        # done condition when the last action was successful inverted
        first_person_obs = self.ctrl.last_event.frame
        # goal = list(self.obj_pos.values())
        # no_collision = self.ctrl.last_event.metadata["lastActionSuccess"]
        # #done = self.get_done(no_collision, visible)

        #   agent position
        # agent_rotation = np.array(list(self.ctrl.last_event.metadata["agent"]["rotation"].values()))
        # agent_pose = np.concatenate((agent_position, agent_rotation), axis=0)
        #
        # #   done when the current agent position equals given goal position
        # done = np.equal(self.goal, agent_position) or not no_collision
        # reward = float("%3f" % (reward))


        return first_person_obs, agent_position, reward, done

    def get_reachable_position(self):
        self.ctrl.step(dict(action='GetReachablePositions'))
        return pd.DataFrame(self.ctrl.last_event.metadata["reachablePositions"]).values

    def action_sampler(self):
        return np.random.choice(self.action_n)

    def get_last_object_distance(self):
        for obj in self.ctrl.last_event.metadata["objects"]:
            if self.object_name in obj["name"]:
                obj_ = obj
        visible = obj_["visible"]
        obj_agent_dis = obj_["distance"]
        return obj_agent_dis, visible

    def get_done(self, no_collision, visible):
        if no_collision and visible:
            done = True
        elif not no_collision:
            done = True
        else:
            done = False
        return done

    def get_random_object(self):
        obj_event = self.ctrl.last_event.metadata["objects"]
        obj = random.sample(obj_event, 1)[0]
        obj_pos = obj["position"]
        visible = obj["visible"]
        obj_agent_dis = obj["distance"]
        return obj_pos, visible, obj_agent_dis

