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

env.make(controller=cl)

agent_pose, done, object_position, _ = env.reset(controller=cl, grid_size=0.15)
print("agent_position:", agent_pose, "object position:", object_position)

done = False
while True:
    action = np.random.randint(0, 5)
    agent_pose_, done, reward, obj_agent_dis, visible, _ = env.take_action(action, controller=cl, object_name="Television")
    print("visible:", visible, "\n agent_pose:", agent_pose_, "\n done:", done, "\n reward:", reward,
          "\n distance:", obj_agent_dis)
    # plt.imshow(frame)
    # plt.show()
