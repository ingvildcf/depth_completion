import glob
import os
import numpy as np
from ip_basic.depth_map_utils import *
from ip_basic.vis_utils import *
from PIL import Image
import time

import matplotlib.pyplot as plt
import pickle as pkl

MAX_DEPTH = 10.00
DEPTH_PARAM1 = 351.30
DEPTH_PARAM2 = 1092.50
EMPTY = 0

N_ROWS = 480
N_COLS = 640

pickle_counter = 600

# Best settings so far: 
fill_type = 'fast'
extrapolate = True
blur_type = None

# Fast fill kernels
custom_kernel = DIAMOND_KERNEL_7
morph_kernel = FULL_KERNEL_5
dilation_kernel = FULL_KERNEL_7

def convert_to_metres(depth_image):
    return DEPTH_PARAM1 / (DEPTH_PARAM2 - depth_image)

def read_pgm(filename):

    """Return a raster of integers from a PGM as a list of lists."""
    raster = []
    try:
        with open(filename, 'rb') as pgmf:
            header = pgmf.readline()
            assert header[:2] == b'P5'
            (width, height) = [int(i) for i in header.split()[1:3]]
            depth = int(header.split()[3])
            assert depth <= 65535
            for y in range(height):
                row = []
                for y in range(width):
                    low_bits = ord(pgmf.read(1))
                    row.append(low_bits+255*ord(pgmf.read(1)))
                raster.append(row)
    except:
        print("Error reading file: {}".format(filename))
        return None
    return np.array(raster)

class SingleImageProcessSequence:
    def __init__(self, projected_depths, completed_depths, completed_depths_2, filled_depths):
        self.projected_depths = projected_depths
        self.completed_depths = completed_depths
        self.completed_depths_2 = completed_depths_2
        self.filled_depths = filled_depths

def find_empties(image):
    empties = []
    indices = (image == EMPTY).nonzero()
    if len(indices[0]) > 0:
        for i in range(len(indices[0])):
            empties.append([indices[0][i], indices[1][i]])
    print('Found: ', len(empties), ' empties.')
    return empties

def fill_empty_spaces(image):
    
    filled_image = image.copy()
    empties = find_empties(image)
    for i in range (len(empties)):
        r = empties[i][0]
        c = empties[i][1]
        nearest = nearest_nonzero_idx(image, r, c)
        filled_image[r][c] = image[nearest[0], nearest[1]]
    return filled_image

def nearest_nonzero_idx(a,x,y):
    idx = np.argwhere(a)
    # If (x,y) itself is also non-zero, we want to avoid those, so delete that
    # But, if we are sure that (x,y) won't be non-zero, skip the next step
    #idx = idx[~(idx == [x,y]).all(1)]

    return idx[((idx - [x,y])**2).sum(1).argmin()]


def process_images(depth_image_list):
    # Declare global variables from above snippet
    global MAX_DEPTH
    global DEPTH_PARAM1
    global DEPTH_PARAM2
    global fill_type
    global extrapolate
    global blur_type
    global custom_kernel
    global morph_kernel
    global dilation_kernel
    global pickle_counter

    image_array = []
    image_num = 0
    for depth_path in depth_image_list:
        print(image_num)
        image_num += 1
        
        try:
            # Read depth image
            depth_image = read_pgm(depth_path)
            # Convert to metres
            projected_depths = convert_to_metres(depth_image)
            projected_depths = np.clip(projected_depths, 0, MAX_DEPTH - 0.03)
            projected_depths = (projected_depths).astype(np.float32)
            # Fill in
            if fill_type == 'fast':
                completed_depths = fill_in_fast(
                    projected_depths, max_depth=MAX_DEPTH, extrapolate=extrapolate, blur_type=blur_type, 
                    morph_kernel=morph_kernel, dilation_kernel=dilation_kernel)

                fill_type = 'fast'
                extrapolate = True
                blur_type = None

                custom_kernel = DIAMOND_KERNEL_9
                morph_kernel = FULL_KERNEL_9
                dilation_kernel = FULL_KERNEL_9

                completed_depths_2 = fill_in_fast(
                    completed_depths, max_depth=MAX_DEPTH, extrapolate=extrapolate, blur_type=blur_type, 
                    morph_kernel=morph_kernel, dilation_kernel=dilation_kernel)
                print('filling...')
                filled_depths = fill_empty_spaces(completed_depths_2)
                print('filled')
                projected_depths = np.ones([N_ROWS, N_COLS])*MAX_DEPTH - projected_depths
                projected_depths[projected_depths==EMPTY] = MAX_DEPTH

                completed_depths = np.ones([N_ROWS, N_COLS])*MAX_DEPTH - completed_depths
                completed_depths[completed_depths==EMPTY] = MAX_DEPTH

                completed_depths_2[completed_depths_2==EMPTY] = MAX_DEPTH

                #filled_depths = np.ones([N_ROWS, N_COLS])*MAX_DEPTH - filled_depths

                image_array.append(SingleImageProcessSequence(projected_depths, completed_depths, completed_depths_2, filled_depths))
            else:
                raise ValueError('Invalid fill_type {}'.format(fill_type))
            
        except:
            print("Error processing file: {}".format(depth_path))
            continue
    
    # save image array as a pickle
    pickle_counter += 1
    path = 'D:/pickles_3/'
    pickle_filename = path + 'pickle_3_' + str(pickle_counter) + '.pkl'

    with open(pickle_filename, 'wb') as f:
        pkl.dump(image_array, f)
        print("Dumped one pickle")


# List all file paths of all files with .pgm extension in a list recursively
# depth_image_paths = glob.glob('/media/mihir/MKSSD/misc_offices_playroom_reception_studies_study_rooms/**/*.pgm', recursive=True)
# depth_image_paths = glob.glob('/media/mihir/MKSSD/cafe_dining_furniture_home_off_kitchen_libraries_living_rooms/**/*.pgm', recursive=True)
#depth_image_paths = glob.glob('D:\misc_offices_playroom_reception_studies_study_rooms/**/*.pgm', recursive=True)
#depth_image_paths = glob.glob('D:/basements_bedrooms_bathrooms_bookstores/**/*.pgm', recursive=True)
#depth_image_paths = glob.glob('D:\cafe_dining_furniture_home_off_kitchen_libraries_living_rooms/**/*.pgm', recursive=True) # 3
#depth_image_paths = glob.glob('D:\cafe_dining_furniture_home_off_kitchen_libraries_living_rooms/home_offices/**/*.pgm', recursive=True) # 3_100
#depth_image_paths = glob.glob('D:\cafe_dining_furniture_home_off_kitchen_libraries_living_rooms/kitchens_part1/**/*.pgm', recursive=True) # 3_200
#depth_image_paths = glob.glob('D:\cafe_dining_furniture_home_off_kitchen_libraries_living_rooms/kitchens_part2/**/*.pgm', recursive=True) # 3_300
#depth_image_paths = glob.glob('D:\cafe_dining_furniture_home_off_kitchen_libraries_living_rooms/libraries/**/*.pgm', recursive=True) # 3_400
#depth_image_paths = glob.glob('D:\cafe_dining_furniture_home_off_kitchen_libraries_living_rooms/living_rooms_part1/**/*.pgm', recursive=True) # 3_500
depth_image_paths = glob.glob('D:\cafe_dining_furniture_home_off_kitchen_libraries_living_rooms/living_rooms_part2/**/*.pgm', recursive=True) # 3_600


#depth_image_paths = glob.glob('D:\misc_offices_playroom_reception_studies_study_rooms\misc_part1\computer_lab_0001/*.pgm', recursive=True)
#depth_image_paths = glob.glob('D:\cafe_dining_furniture_home_off_kitchen_libraries_living_rooms\living_rooms_part2\living_room_0031/*.pgm', recursive=True)


tenth_of_depth_images = depth_image_paths[::10]
#tenth_of_depth_images = depth_image_paths[:2]

""""""
# Save file paths in text file
with open('image_file_paths_3_600.txt', mode='wt', encoding='utf-8') as path_file: ###########################################################################
    path_file.write('\n'.join(tenth_of_depth_images))

size_per_pickle = 1000
list_of_lists = [tenth_of_depth_images[i:i + size_per_pickle] for i in range(0, len(tenth_of_depth_images), size_per_pickle)]
print('Start', list_of_lists[0][0])
print("Number of lists: {}".format(len(list_of_lists)))

print('Number of depth images: {}'.format(len(depth_image_paths)))
print('Number of depth images to process: {}'.format(len(tenth_of_depth_images)))

#process_images(list_of_lists[0])

for i in range(0, len(list_of_lists)): ################################################################################################################# CHANGE THIS 
    print("Global counter: {}".format(pickle_counter))
    print("Processing list {}".format(i))
    process_images(list_of_lists[i])


#process_images(tenth_of_depth_images)

plotting = None
if plotting is not None:

    depth_objects = []
    with (open("pickle_3_" + str(pickle_counter) + ".pkl", "rb")) as pickle_file:
        depth_objects.append(pkl.load(pickle_file))

    proj_1 = depth_objects[0][0].projected_depths
    compl_11 = depth_objects[0][0].completed_depths
    compl_12 = depth_objects[0][0].completed_depths_2
    filled_1 = depth_objects[0][0].filled_depths


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