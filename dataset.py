import numpy as np
import random
import ai2thor.controller
from Environment_dataset_generation import Environment

_BATCH_SIZE = 20  # 16
_NUM_BATCHES = 10  # 16
_TIME_STEPS = 250  # 150

def generate_scene_name():
    scene_name_list = []
    for k in range(430):
        if k in range(1, 31):
            scene_name_list.append("FloorPlan" + str(k))
        elif k in range(201, 231):
            scene_name_list.append("FloorPlan" + str(k))
        elif k in range(301, 331):
            scene_name_list.append("FloorPlan" + str(k))
        elif k in range(400, 431):
            scene_name_list.append("FloorPlan" + str(k + 1))

    return scene_name_list


env = Environment(top_view_cam=False)

env.make()

i = 0
for scene in generate_scene_name():
    x = scene[9:]
    print("scene:", x)
    i += 1
    obs_data = []
    for i_episode in range(_BATCH_SIZE):
        observation, agent_position, agent_pose, _, _, _ = env.reset(scene=scene, random_init=True)
        print("agent_init_pos", agent_position)
        obs_data.append(observation)

        for _ in range(_TIME_STEPS):
            action = np.random.randint(0, 6)
            observation, _, _, _ = env.take_action(action)
            obs_data.append(observation)

    if int(x) in range(1, 21):
        np.save('/Volumes/WIN/Kitchens/Bathrooms/VAE_{}'.format(scene), obs_data)
        obs_data = None

    elif int(x) in range(21, 28):
        np.save('/Volumes/WIN/Kitchens/validation/VAE_{}'.format(scene), obs_data)
        obs_data = None

    elif int(x) in range(28, 31):
        np.save('/Volumes/WIN/Kitchens/testing/VAE_{}'.format(scene), obs_data)
        obs_data = None

    elif int(x) in range(201, 221):
        np.save('/Volumes/WIN/LivingRoom/Bathrooms/VAE_{}'.format(scene), obs_data)
        obs_data = None

    elif int(x) in range(221, 228):
        np.save('/Volumes/WIN/LivingRoom/validation/VAE_{}'.format(scene), obs_data)
        obs_data = None

    elif int(x) in range(228, 231):
        np.save('/Volumes/WIN/LivingRoom/testing/VAE_{}'.format(scene), obs_data)
        obs_data = None

    elif int(x) in range(301, 321):
        np.save('/Volumes/WIN/Bedrooms/Bathrooms/VAE_{}'.format(scene), obs_data)
        obs_data = None

    elif int(x) in range(321, 328):
        np.save('/Volumes/WIN/Bedrooms/validation/VAE_{}'.format(scene), obs_data)
        obs_data = None

    elif int(x) in range(328, 331):
        np.save('/Volumes/WIN/Bedrooms/testing/VAE_{}'.format(scene), obs_data)
        obs_data = None

    elif int(x) in range(401, 421):
        np.save('/Volumes/WIN/Bathrooms/Bathrooms/VAE_{}'.format(scene), obs_data)
        obs_data = None

    elif int(x) in range(421, 428):
        np.save('/Volumes/WIN/Bathrooms/validation/VAE_{}'.format(scene), obs_data)
        obs_data = None

    elif int(x) in range(428, 431):
        np.save('/Volumes/WIN/Bathrooms/testing/VAE_{}'.format(scene), obs_data)
        obs_data = None
