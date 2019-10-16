#TODO   make it much easier to handle
#TODO   extend the Controller class of ai2thor


from ai2thor.controller import Controller
from ai2thor.controller import BFSController
import ai2thor.controller
import numpy as np


#    This axis has “right hand” facing with respect to the forward Z-Axis,
#    Y-axis pointing upward, z-axis pointing forward, x axis  pointing to the left

class Environment(object):

    def __int__(self):
        # self.controller = Controller()
        # self.start = self.controller.start()
        # self.reset = self.controller.reset()
        # self.step = self.controller.step()
        pass

    @classmethod
    def make(cls, controller, start_unity=True, scene="FloorPlan220", player_screen_width=300,
             player_screen_height=300, top_view_camera=True):

        controller.start(start_unity=start_unity, player_screen_width=player_screen_width,
                         player_screen_height=player_screen_height)

        controller.reset(scene_name=scene)

        controller.step(dict(action="TeleportFull", x=-4.25, y=0.909619451, z=2.75, rotation=90, horizon=0.0))

        if top_view_camera:
            event = controller.step(dict(action="ToggleMapView"))

    @classmethod
    def reset(cls, controller, grid_size=0.25, depth_image=False, class_image=False, object_image=False,
              visibility_distance=1.5, camera_Y=0.675, fov=60.0, object_name="LightSwitch"):

        event = controller.step(dict(action="Initialize", gridSize=grid_size, renderDepthImage=depth_image,
                             renderClassImage=class_image, renderObjectImage=object_image,
                             visibilityDistance=visibility_distance, cameraY=camera_Y, fieldOfView=fov))

        #TODO generalize the position of the thirdPartyCamera to all environment
        third_party_cam = controller.last_event.metadata["thirdPartyCameras"]
        if len(third_party_cam) == 0:
            event = controller.step(
                dict(action='AddThirdPartyCamera', rotation=dict(x=90, y=0, z=0),
                        position=dict(x=-2.847458, y=7.5, z=1.9)))

        objects = event.metadata["objects"]
        for obj in objects:
            if object_name in obj["name"]:
                obj = obj
        object_position = obj["position"]
        agent_obj_distace = obj["distance"]
        obs = event.third_party_camera_frames
        frame = np.squeeze(obs, axis=0)
        done = event.metadata["lastActionSuccess"]
        agent_position = list(event.metadata["agent"]["position"].values())
        camera_pose = event.metadata["cameraPosition"]
        # agent_rotation = np.array(list(event.metadata["agent"]["rotation"].values()), dtype=float)
        # object_position = list(obj["position"].values())
        # agent_pose = np.concatenate((agent_position, agent_rotation), axis=0)
        goal = [-0.075, 0.909619451, 1.75]
        return frame, agent_position, done, goal, object_position

    @classmethod
    def take_action(cls, action, controller, object_name="LightSwitch"):
        reward = 0
        if action == 0:
            event = controller.step(dict(action="MoveRight"))
            objects = event.metadata["objects"]
            for obj in objects:
                if object_name in obj["name"]:
                    obj = obj
            visible = obj["visible"]
            obj_agent_dis = obj["distance"]
            if event.metadata["lastActionSuccess"]:
                reward = 0
                if visible:
                    reward = 1
            else:
                reward = -1
        elif action == 1:
            event = controller.step(dict(action='RotateRight'))
            objects = event.metadata["objects"]
            for obj in objects:
                if object_name in obj["name"]:
                    obj = obj
            visible = obj["visible"]
            obj_agent_dis = obj["distance"]
            if event.metadata["lastActionSuccess"]:
                reward = 0
                if visible:
                    reward = 1
            else:
                reward = -1
        elif action == 2:
            event = controller.step(dict(action="RotateLeft"))
            objects = event.metadata["objects"]
            for obj in objects:
                if object_name in obj["name"]:
                    obj = obj
            visible = obj["visible"]
            obj_agent_dis = obj["distance"]
            if event.metadata["lastActionSuccess"]:
                reward = 0
                if visible:
                    reward = 1
            else:
                reward = -1

        elif action == 3:
            event = controller.step(dict(action='MoveLeft'))
            objects = event.metadata["objects"]
            for obj in objects:
                if object_name in obj["name"]:
                    obj = obj
            visible = obj["visible"]
            obj_agent_dis = obj["distance"]
            if event.metadata["lastActionSuccess"]:
                reward = 0
                if visible:
                    reward = 1
            else:
                reward = -1

        elif action == 4:
            event = controller.step(dict(action="MoveAhead"))
            objects = event.metadata["objects"]
            for obj in objects:
                if object_name in obj["name"]:
                    obj = obj
            visible = obj["visible"]
            obj_agent_dis = obj["distance"]
            if event.metadata["lastActionSuccess"]:
                reward = 0
                if visible:
                    reward = 1
            else:
                reward = -1

        else:
            event = controller.step(dict(action="MoveBack"))
            objects = event.metadata["objects"]
            for obj in objects:
                if object_name in obj["name"]:
                    obj = obj
            visible = obj["visible"]
            obj_agent_dis = obj["distance"]
            if event.metadata["lastActionSuccess"]:
                reward = 0
                if visible:
                    reward = 1
            else:
                reward = -1

        # obs = event.frame
        obs = event.third_party_camera_frames
        frame = np.squeeze(obs, axis=0)
        done = event.metadata["lastActionSuccess"]
        agent_position = list(event.metadata["agent"]["position"].values())
        agent_rotation = list(event.metadata["agent"]["rotation"].values())
        # agent_pose = np.concatenate((agent_position, agent_rotation), axis=0)

        return frame, agent_position, done, reward, obj_agent_dis, visible

    @classmethod
    def reward(cls, obj_agent_dis_, obj_agent_dis):
        if obj_agent_dis_ < obj_agent_dis:
            reward = 1
        else:
            reward = 0
        return reward