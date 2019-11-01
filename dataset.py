import numpy as np
import re
from Environment import Environment
from PIL import Image
import h5py


_BATCH_SIZE = 1#16
_NUM_BATCHES = 1#16
_TIME_STEPS = 1#150
_RENDER = True


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
            scene_name_list.append("FloorPlan" + str(k+1))

    return scene_name_list


env = Environment(top_view_cam=False)

env.make()
obs_data = []
i = 0
for scene in generate_scene_name():
        x = scene[9:]
        i+=1
        for i_episode in range(_BATCH_SIZE):
            observation = env.reset(scene=scene)

            obs_sequence = []

            for _ in range(_TIME_STEPS):
                action = np.random.randint(0, 5)

                observation, _, _, _, _ = env.take_action(action)
                if int(x) in range(1,21):
                    print("kitchens_training")
                    obs_data.append(observation)
                elif int(x) in range(21, 28):
                    print("kitchens_validation")
                    print(x)
                elif int(x) in range(28, 31):
                    print("kitchens_testing")
                    print(x)

        #         elif int(x) in range(201, 221):
        #             print("Living_training")
        #             print(x)
        #         elif int(x) in range(221, 228):
        #             print("Living_validation")
        #             print(x)
        #         elif int(x) in range(228, 231):
        #             print("Living_testing")
        #             print(x)
        #
        #         elif int(x) in range(301, 321):
        #             print("Bedrooms_training")
        #             print(x)
        #         elif int(x) in range(321, 328):
        #             print("Bedrooms_validation")
        #             print(x)
        #         elif int(x) in range(328, 331):
        #             print("Bedrooms_testing")
        #             print(x)
        #
        #         elif int(x) in range(401, 421):
        #             print("LBathrooms_training")
        #             print(x)
        #         elif int(x) in range(421, 428):
        #             print("Bathrooms_validation")
        #             print(x)
        #         elif int(x) in range(428, 431):
        #             print("Bathrooms_testing")
        #             print(x)
        # print(i)


