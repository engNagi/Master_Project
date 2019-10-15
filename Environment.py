from ai2thor.controller import Controller
import ai2thor.controller
import numpy as np


#    This axis has “right hand” facing with respect to the forward Z-Axis,
#    Y-axis pointing upward, z-axis pointing forward, x axis  pointing to the left

class Environment(object):

    def __int__(self):
        pass

    @classmethod
    def make(cls, controller, start_unity=True, scene="FloorPlan230", player_screen_width=300,
             player_screen_height=300, top_view_camera=False):

        controller.start(start_unity=start_unity, player_screen_width=player_screen_width,
                         player_screen_height=player_screen_height)

        controller.reset(scene_name=scene)

        controller.step(dict(action="TeleportFull", x=-4.5, y=0.9082557, z=3.75, rotation=0.0, horizon=0.0))

        if top_view_camera:
            controller.step(dict(action="ToggleMapView"))

    @classmethod
    def reset(cls, controller, grid_size=0.25, depth_image=False, class_image=False, object_image=False,
              visibility_distance=1.5, camera_Y=0.675, fov=60.0, object_name="LightSwitch"):

        event = controller.step(dict(action="Initialize", gridSize=grid_size, renderDepthImage=depth_image,
                                     renderClassImage=class_image, renderObjectImage=object_image,
                                     visibilityDistance=visibility_distance, cameraY=camera_Y, fieldOfView=fov))

        objects = event.metadata["objects"]
        for obj in objects:
            if object_name in obj["name"]:
                obj = obj
        object_position = obj["position"]
        obs = event.frame
        done = not event.metadata["lastActionSuccess"]
        agent_position = np.array(list(event.metadata["agent"]["position"].values()), dtype=float)
        agent_rotation = np.array(list(event.metadata["agent"]["rotation"].values()), dtype=float)
        object_position = np.array(list(obj["position"].values()), dtype=float)
        agent_pose = np.concatenate((agent_position, agent_rotation), axis=0)
        return agent_pose, done, object_position, obs

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
            event = controller.step(dict(action='RotateLeft'))
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

        obs = event.frame
        done = not event.metadata["lastActionSuccess"]
        agent_position = np.array(list(event.metadata["agent"]["position"].values()), dtype=float)
        agent_rotation = np.array(list(event.metadata["agent"]["rotation"].values()), dtype=float)
        agent_pose = np.concatenate((agent_position, agent_rotation), axis=0)

        return agent_pose, done, reward, obj_agent_dis, visible, obs
