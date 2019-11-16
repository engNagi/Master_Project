import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


x = np.load("/Volumes/WIN/Kitchens/training/VAE_FloorPlan1.npy")
for xs in x:
    imgplot = plt.imshow(xs)
    plt.show()

