from Environment_dataset_generation import Environment
import numpy as np
import matplotlib.pyplot as plt

env = Environment(top_view_cam=False, full_scrn=True)

env.make()
first_person_obs, agent_position, goal, _,_, _ = env.reset("FloorPlan220")

done = False
while not done:
    action = np.random.randint(0, 6)
    print(action)
    frame, agent_position, done, reward = env.take_action(action)
    print(env.get_reachable_position())
    print("agent_pose:", agent_position, "\n done:", done, "goal:", goal)
    plt.imshow(frame)
    plt.show()
