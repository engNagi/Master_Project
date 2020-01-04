from Environment import Environment
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import ai2thor.controller
controller = ai2thor.controller.Controller()
controller.start()
controller.reset(scene_name='FloorPlan220')
reachable = controller.step(dict(action="GetReachablePositions"))
controller.step(dict(action='TeleportFull', x=-0.9, y=0.900998235, z=3.50, rotation=180))
#controller.step(dict(action='LookDown'))
event = controller.step(dict(action='Rotate', rotation=180))
# In FloorPlan28, the agent should now be looking at a mug
for o in event.metadata['objects']:
    if o['visible'] and o['pickupable'] and o['objectType'] == 'Mug':
        event = controller.step(dict(action='PickupObject', objectId=o['objectId']), raise_for_failure=True)
        mug_object_id = o['objectId']
        break
