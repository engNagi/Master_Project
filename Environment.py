from ai2thor.controller import Controller
import ai2thor.controller
import numpy as np


class Environment(object):

    def __int__(self):
        pass

    @classmethod
    def reset(cls, controller, start_unity=True, scene="FloorPlan230", depth_image=False,
              grid_size=0.25, class_image=False, Object_image=False, visibility_distance=1.5, camera_Y=0.675,
              fov=60.0, player_screen_width=300, player_screen_height=300, top_view_camera=True):

        controller.start(start_unity=start_unity, player_screen_width=player_screen_width,
                         player_screen_height=player_screen_height)

        controller.reset(scene_name=scene)

        event = controller.step(dict(action="Initialize", gridSize=grid_size, renderDepthImage=depth_image,
                                     renderClassImage=class_image, renderObjectImage=Object_image,
                                     visibilityDistance=visibility_distance, cameraY=camera_Y, fieldOfView=fov))

        controller.step(dict(action="TeleportFull", x=-4.5, y=0.9082557, z=3.75, rotation=0.0, horizon=0.0))

        if top_view_camera:
            controller.step(dict(action="ToggleMapView"))

        not_done = event.metadata["lastActionSuccess"]
        agent_position = np.array(list(event.metadata["agent"]["position"].values()), dtype=float)
        agent_rotation = np.array(list(event.metadata["agent"]["rotation"].values()), dtype=float)
        agent_pose = np.concatenate((agent_position, agent_rotation), axis=0)
        return agent_pose, event, not_done, event.frame

    @classmethod
    def is_visible(cls, event):
        objects = event.metadata["objects"]
        for obj in objects:
            if "LightSwitch" in obj["name"]:
                 visible = obj["visible"]
        return visible

    @classmethod
    def take_action(cls, action, controller, visible):
        reward =0
        if action == 0:
            event = controller.step(dict(action="MoveRight"))
            if event.metadata["lastActionSuccess"]:
                reward = 0
                if visible:
                    reward = 1
            else:
                reward = -1
        elif action == 1:
            event = controller.step(dict(action='RotateRight'))
            if event.metadata["lastActionSuccess"]:
                reward = 0
                if visible:
                    reward = 1
            else:
                reward = -1
        elif action == 2:
            event = controller.step(dict(action="RotateLeft"))
            if event.metadata["lastActionSuccess"]:
                reward = 0
                if visible:
                    reward = 1
            else:
                reward = -1

        elif action == 3:
            event = controller.step(dict(action='RotateLeft'))
            if event.metadata["lastActionSuccess"]:
                reward = 0
                if visible:
                    reward = 1
            else:
                reward = -1

        elif action == 4:
            event = controller.step(dict(action="MoveAhead"))
            if event.metadata["lastActionSuccess"]:
                reward = 0
                if visible:
                    reward = 1
            else:
                reward = -1

        else:
            event = controller.step(dict(action="MoveBack"))
            if event.metadata["lastActionSuccess"]:
                reward = 0
                if visible:
                    reward = 1
            else:
                reward = -1

        obs = event.frame
        not_done = event.metadata["lastActionSuccess"]
        agent_position = np.array(list(event.metadata["agent"]["position"].values()), dtype=float)
        agent_rotation = np.array(list(event.metadata["agent"]["rotation"].values()), dtype=float)
        agent_pose = np.concatenate((agent_position, agent_rotation), axis=0)
        return agent_pose, not_done, reward, obs
