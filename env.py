from Environment import Environment as env
import numpy as np
import ai2thor.controller
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cl = ai2thor.controller.Controller(fullscreen=True)
# cl.start()
# # can be any one of the scenes FloorPlan###
# #LightSwitch
# cl.reset('FloorPlan28')
# event = cl.step(dict(action='Initialize', gridSize=0.25))


agent_pose, event, not_done, _ = env.reset(controller=cl, top_view_camera=True)
print("agent_position:", agent_pose)

#done = false
while not_done:
    action = np.random.randint(0, 5)
    visible = env.is_visible(event=event)
    agent_pose_, not_done, reward, _ = env.take_action(action, controller=cl, visible=visible)
    print("visible:", visible, "agent_pose:", agent_pose_, "not done:", not_done, "reward:",reward )
    # plt.imshow(frame)
    # plt.show()
