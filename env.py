from Environment import Environment
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



env = Environment(top_view_cam=False)

env.make()
first_person_obs, agent_position, object_name, object_position, object_agent_dis = env.reset()

done = False
while not done:
    action = np.random.randint(0, 5)
    frame, agent_position, done, reward = env.take_action(action)
    print("agent_pose:", agent_position, "\n done:", done, "goal:", object_name)
    im = Image.fromarray(frame)
    im.save("frame.jpeg")
    plt.imshow(im)
    plt.show()
