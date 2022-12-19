import pickle as pkl
import matplotlib.pyplot as plt

from ip_basic.depth_map_utils import *
from ip_basic.vis_utils import *

MAX_DEPTH = 10.00
EMPTY = 0

N_ROWS = 480
N_COLS = 640

class SingleImageProcessSequence:
    def __init__(self, projected_depths, completed_depths, completed_depths_2, filled_depths):
        self.projected_depths = projected_depths
        self.completed_depths = completed_depths
        self.completed_depths_2 = completed_depths_2
        self.filled_depths = filled_depths

depth_objects = []
path = "C:\\Users\\ingvilcf\\OneDrive - NTNU\Documents\\Master\Data\\depth_completion_python\\depth_completion\\2moprss"
with (open(path + "\\pickle_2_1.pkl", "rb")) as pickle_file:
    depth_objects.append(pkl.load(pickle_file))

path = 'D:\\misc_offices_playroom_reception_studies_study_rooms\offices_part1\\nyu_office_0\d-1294439166.435284-3954546877.pgm'
im_number = 0 

plt.imshow(path)

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


proj_1 = depth_objects[0][im_number].projected_depths
compl_11 = depth_objects[0][im_number].completed_depths

# Best settings so far: 
fill_type = 'fast'
extrapolate = True
blur_type = None

# Fast fill kernels
custom_kernel = DIAMOND_KERNEL_7
morph_kernel = FULL_KERNEL_5
dilation_kernel = FULL_KERNEL_7

projected_depths = proj_1.copy()
projected_depths[projected_depths==MAX_DEPTH] = EMPTY
#projected_depths =  np.ones([N_ROWS, N_COLS])*MAX_DEPTH - projected_depths
projected_depths = (projected_depths).astype(np.float32)

completed_depths = fill_in_fast(
    projected_depths, max_depth=MAX_DEPTH, extrapolate=extrapolate, blur_type=blur_type, 
    morph_kernel=morph_kernel, dilation_kernel=dilation_kernel)

completed_depths[completed_depths==EMPTY] = MAX_DEPTH
# Plot the subplots

# Plot 1
plt.subplot(1, 2, 1)
plt.imshow(proj_1, cmap='gray', vmin=0, vmax=MAX_DEPTH)

# Plot 2
plt.subplot(1, 2, 2)
plt.imshow(completed_depths, cmap='gray', vmin=0, vmax=MAX_DEPTH)

# Plot 3
#plt.subplot(1, 3, 3)
#plt.imshow(compl_11, cmap='gray', vmin=0, vmax=MAX_DEPTH)
#plt.imshow(projected_depths, cmap='gray', vmin=0, vmax=MAX_DEPTH)

plt.show()