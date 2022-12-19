import pickle as pkl
import matplotlib.pyplot as plt

MAX_DEPTH = 10

class SingleImageProcessSequence:
    def __init__(self, projected_depths, completed_depths, completed_depths_2, filled_depths):
        self.projected_depths = projected_depths
        self.completed_depths = completed_depths
        self.completed_depths_2 = completed_depths_2
        self.filled_depths = filled_depths

depth_objects = []
path = "C:\\Users\\ingvilcf\\OneDrive - NTNU\Documents\\Master\Data\\depth_completion_python\\depth_completion\\2moprss"
with (open(path + "\\pickle_2_3.pkl", "rb")) as pickle_file:
    depth_objects.append(pkl.load(pickle_file))


im_number_1 = 900
im_number_2 = 400
im_number_3 = 600 
im_number_4 = 100
#### 2_1
# 400 = coridoor
# 000 = chair 

#### 2_2
# 400 = coridoor 
# 600 = coridoor + bars


#### 2_3
# 900 = desk and two chairs
# 400 
# 600 = desk, chair, lamp
# 100 = ladder thingy?



proj_1 = depth_objects[0][im_number_1].projected_depths
compl_11 = depth_objects[0][im_number_1].completed_depths
compl_12 = depth_objects[0][im_number_1].completed_depths_2
filled_1 = depth_objects[0][im_number_1].filled_depths

proj_2 = depth_objects[0][im_number_2].projected_depths
compl_21 = depth_objects[0][im_number_2].completed_depths
compl_22 = depth_objects[0][im_number_2].completed_depths_2
filled_2 = depth_objects[0][im_number_2].filled_depths

proj_3 = depth_objects[0][im_number_3].projected_depths
compl_31 = depth_objects[0][im_number_3].completed_depths
compl_32 = depth_objects[0][im_number_3].completed_depths_2
filled_3 = depth_objects[0][im_number_3].filled_depths

proj_4 = depth_objects[0][im_number_4].projected_depths
compl_41 = depth_objects[0][im_number_4].completed_depths
compl_42 = depth_objects[0][im_number_4].completed_depths_2
filled_4 = depth_objects[0][im_number_4].filled_depths

#plotting = 1 # Process
plotting = 2 # Comparison

if plotting == 1:
    # Plot the subplots
    # Plot 1
    plt.subplot(2, 2, 1)
    #plt.plot(x, y1, 'g')
    plt.imshow(proj_1, cmap='gray', vmin=0, vmax=MAX_DEPTH)
    # Plot 2
    plt.subplot(2, 2, 2)
    #plt.plot(x, y2, '-.r')
    plt.imshow(compl_11, cmap='gray', vmin=0, vmax=MAX_DEPTH)
    # Plot 3
    plt.subplot(2, 2, 3)
    #plt.plot(x, y3, ':y')
    plt.imshow(compl_12, cmap='gray', vmin=0, vmax=MAX_DEPTH)
    # Plot 4
    plt.subplot(2, 2, 4)
    #plt.plot(x, y4, '--c')
    plt.imshow(filled_1, cmap='gray', vmin=0, vmax=MAX_DEPTH)
    plt.show()

if plotting == 2:
    # Plot the subplots
    # Plot 1
    plt.subplot(2, 2, 1)
    #plt.plot(x, y1, 'g')
    plt.imshow(proj_1, cmap='gray', vmin=0, vmax=MAX_DEPTH)
    # Plot 2
    plt.subplot(2, 2, 2)
    #plt.plot(x, y2, '-.r')
    plt.imshow(proj_2, cmap='gray', vmin=0, vmax=MAX_DEPTH)
    # Plot 3
    plt.subplot(2, 2, 3)
    #plt.plot(x, y3, ':y')
    plt.imshow(proj_3, cmap='gray', vmin=0, vmax=MAX_DEPTH)
    # Plot 4
    plt.subplot(2, 2, 4)
    #plt.plot(x, y4, '--c')
    plt.imshow(proj_4, cmap='gray', vmin=0, vmax=MAX_DEPTH)
    plt.show()