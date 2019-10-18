from Environment import Environment as env
import numpy as np
import ai2thor.controller
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cl = ai2thor.controller.Controller(fullscreen=True)
# cl.start()
# # can be any one of the scenes FloorPlan###
# #LightSwitch
# cl.search_all_closed('FloorPlan220')
# print(cl.grid_points)
# cl.reset('FloorPlan28')
# event = cl.step(dict(action='Initialize', gridSize=0.25))
# reachable_positions = event.metadata['reachablePositions']
#
#
env.make(controller=cl)

agent_pose, done, reward, agent_obj_distance, object_position, obs = env.reset( controller=cl, grid_size=0.15,
                                                                       object_name="Television")
print("agent_position:", agent_pose, "object position:", object_position)
plt.imshow(obs)
plt.show()

done =True
while done:
    action = np.random.randint(0, 5)
    agent_pose_, done, reward, obj_agent_dis, visible, obs_ = env.take_action(action, controller=cl, object_name="Television")
#     print("visible:", visible, "\n agent_pose:", agent_pose_, "\n done:", done, "\n reward:", reward,
#           "\n distance:", obj_agent_dis)
    plt.imshow(obs_)
    plt.show()
