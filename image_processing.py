import glob
import os
import numpy as np
from ip_basic.depth_map_utils import *
from ip_basic.vis_utils import *
from PIL import Image
import time

from matplotlib import pyplot as plt
import pickle as pkl

MAX_DEPTH    = 10.00
DEPTH_PARAM1 = 351.30
DEPTH_PARAM2 = 1092.50

N_ROWS = 480
N_COLS = 640
EMPTY  = 0

# Best settings so far: 
fill_type   = 'fast'
extrapolate = True
blur_type   = None

# Fast fill kernels
custom_kernel   = DIAMOND_KERNEL_7
morph_kernel    = FULL_KERNEL_5
dilation_kernel = FULL_KERNEL_7

pickle_counter = 0

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
                    row.append(low_bits + 255*ord(pgmf.read(1)))
                raster.append(row)
    except:
        print("Error reading file: {}".format(filename))
        return None
    return np.array(raster)

class SingleImageProcessSequence:
    def __init__(self, projected_depths, completed_depths, filled_depths):
        self.projected_depths = projected_depths
        self.completed_depths = completed_depths
        self.filled_depths = filled_depths

class SingleImageProcess:
    def __init__(self, projected_depths, completed_depths, completed_depths_2, filled_depths):
        self.projected_depths = projected_depths
        self.completed_depths = completed_depths
        self.completed_depths_2 = completed_depths_2
        self.filled_depths = filled_depths

def fill_empty_spaces(image, empties):
    filled_image = image.copy()
    """
    for r in range(0, N_ROWS):
        for c in range(0, N_COLS):
            if [r, c] in empties:
                # find_nearest_valid is a two to three times faster than fid_nearest_valid_2 
                nearest = find_nearest_valid(image, r, c)
                filled_image[r][c] = image[nearest[0], nearest[1]]
    return filled_image
    """
    
    for i in range(len(empties)):
        r = empties[i][0]
        c = empties[i][1]
        print(r, c)
        nearest = nearest_nonzero_idx(image, r, c)
        print(nearest)
        filled_image[r][c] = image[nearest[0], nearest[1]]

# Find nearest known pixel
def find_nearest_valid(image, row, col):
    
    #n_rows = len(image[0])
    #n_cols = len(image)

    # Find distances to closest valid pixel in all directions
    east = N_COLS
    south = N_ROWS
    west = N_COLS
    north = N_ROWS
    
    for e in range(col, N_COLS):
        if image[row, e] > EMPTY:
            east = e - col
            break

    for s in range(row, N_ROWS):
        if image[s, col] > EMPTY:
            south = s - row
            break
    
    for w in range(col, -1, -1):
        if image[row, w] > EMPTY:
            west = col - w
            break
    
    for n in range(row, -1, -1):
        if image[n, col] > EMPTY:
            north = row - n
            break

    valid_neighbor_dist = [east, south, west, north]

    near_idx = valid_neighbor_dist.index(min(valid_neighbor_dist))
    
    if near_idx == 0:
        nearest_valid_pixel = [row, col + east]
    if near_idx == 1:
        nearest_valid_pixel = [row + south, col]
    if near_idx == 2:
        nearest_valid_pixel = [row, col - west]
    if near_idx == 3:
        nearest_valid_pixel = [row - north, col]    
    
    return nearest_valid_pixel

def find_nearest_valid_2(image, row, col):
    idx = np.argwhere(image)
    return idx[((idx - [row,col])**2).sum(1).argmin()]

def nearest_nonzero_idx(a,x,y):
    idx = np.argwhere(a)

    # If (x,y) itself is also non-zero, we want to avoid those, so delete that
    # But, if we are sure that (x,y) won't be non-zero, skip the next step
    idx = idx[~(idx == [x,y]).all(1)]

    return idx[((idx - [x,y])**2).sum(1).argmin()]

# Find all empty pixels
def find_empties(image):

    empties = []
    indices = (image == EMPTY).nonzero()
    if len(indices[0]) > 1:
        for i in range(len(indices)):
            empties.append([indices[0][i], indices[1][i]])
    """
    for r in range(0, N_ROWS):
        for c in range(0, N_COLS):
            # Empty pixel found
            if(image[r][c] == EMPTY):
                # Already found
                if [r, c] in empties:
                    pass
                # New empty
                else:
                    empty_area_indexes.append([r,c])
                    empties = find_empty_area(image, empties, r, c)
    """     
    print(indices)
    print('Found total', len(empties))
    print(empties)            
    return empties

# Finds the empty area connected to pixel [row, col] 
def find_empty_area(image, empties, row, col):
    # Add first empty pixel to emties and to queue
    empties.append([row, col])
    queue = [[row, col]]

    #n_rows = len(image[0])
    #n_cols = len(image)
    
    # Continue until no more new empty neighbor pixels in queue
    while len(queue) > 0:
        # Indexes of first pixel in queue
        r = queue[0][0]
        c = queue[0][1]
       
        # East neighbor
        if c < N_COLS - 1:
            if ([r, c + 1] not in empties) and ([r, c + 1] not in queue) and (image[r, c + 1] == EMPTY):
                empties.append([r, c + 1])
                queue.append([r, c + 1])
        # South neighbor
        if r < N_ROWS - 1:
            if ([r + 1, c] not in empties) and ([r + 1, c] not in queue) and (image[r + 1, c] == EMPTY):
                empties.append([r + 1, c])
                queue.append([r + 1, c])
        
        # For all rows except the top one    
        if r > row:
            # West neighbor
            if c > 0:
                if ([r, c - 1] not in empties) and ([r, c - 1] not in queue) and (image[r, c - 1] == EMPTY):
                    empties.append([r, c - 1])
                    queue.append([r, c - 1])
            # North neighbor
            if ([r - 1, c] not in empties) and ([r - 1, c] not in queue) and (image[r - 1, c] == EMPTY):
                empties.append([r - 1, c])
                queue.append([r - 1, c])
        
        # Remove finished pixel
        queue.remove([r, c])

        # Remove duplicates                                                             WHY IS THIS NECESARRY??
        unique_queue = []
        [unique_queue.append(pixel) for pixel in queue if pixel not in unique_queue]
        queue = unique_queue
    return empties

def fill_empty_depths(completed_depths):
    empty_depths = find_empties(completed_depths)

    filled_depths = fill_empty_spaces(completed_depths, empty_depths)
    return filled_depths

def process_images_old(depth_image_list):
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

    for depth_path in depth_image_list:
        try:
            # Read depth image
            depth_image = read_pgm(depth_path)

            # Convert to metres
            projected_depths = convert_to_metres(depth_image)
            projected_depths = np.clip(projected_depths, 0, MAX_DEPTH - 0.03)
            projected_depths = (projected_depths).astype(np.float32)

            # Fill in depths
            if fill_type == 'fast':
                # Depth completion
                completed_depths = fill_in_fast(
                    projected_depths, max_depth=MAX_DEPTH, extrapolate=extrapolate, blur_type=blur_type, 
                    morph_kernel=morph_kernel, dilation_kernel=dilation_kernel)
                
                # Depth filling
                filled_depths = fill_empty_depths(completed_depths)
                
                # Add process sequence object for single image to array
                image_array.append(SingleImageProcessSequence(projected_depths, completed_depths, filled_depths))
                print('Image ', depth_image, ' saved to pickle. ')
                # image_array.append(final_depths)
            else:
                raise ValueError('Invalid fill_type {}'.format(fill_type))
        except:
            print("Error processing file: {}".format(depth_path))
            continue
    
    # save image array as a pickle
    pickle_counter += 1
    pickle_filename = 'pickle_' + str(pickle_counter) + '.pkl'
    with open(pickle_filename, 'wb') as f:
        pkl.dump(image_array, f)


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
        image_num+=1

        # Read depth image
        depth_image = read_pgm(depth_path)
        # Convert to metres
        projected_depths = convert_to_metres(depth_image)
        projected_depths = np.clip(projected_depths, 0, MAX_DEPTH - 0.03)
        projected_depths = (projected_depths).astype(np.float32)
        
        # Fill in depths
        if fill_type == 'fast':
            # Depth completion
            completed_depths = fill_in_fast(
                projected_depths, max_depth=MAX_DEPTH, extrapolate=extrapolate, blur_type=blur_type, 
                morph_kernel=morph_kernel, dilation_kernel=dilation_kernel)
            #custom_kernel   = DIAMOND_KERNEL_5
            #morph_kernel    = FULL_KERNEL_3
            #dilation_kernel = FULL_KERNEL_5
            completed_depths_2 = fill_in_fast(
                completed_depths, max_depth=MAX_DEPTH, extrapolate=extrapolate, blur_type=blur_type, 
                morph_kernel=morph_kernel, dilation_kernel=dilation_kernel)
            
            # Depth filling
            filled_depths = fill_empty_depths(completed_depths_2)
            
            # Display empty values as white
            #print(len(completed_depths_2[completed_depths_2 == 0]))
            #completed_depths_2[completed_depths_2 == 0] = 1
            
            # Invert and set empty depths to MAX to show as white
            projected_depths = np.ones([N_ROWS, N_COLS])*MAX_DEPTH - projected_depths
            completed_depths = np.ones([N_ROWS, N_COLS])*MAX_DEPTH - completed_depths
            completed_depths[completed_depths == 0] = MAX_DEPTH
            completed_depths_2[completed_depths_2 == 0] = MAX_DEPTH
            # Add process sequence object for single image to array
            image_array.append(SingleImageProcess(projected_depths, completed_depths, completed_depths_2, filled_depths))
            print('Image ', os.path.basename(depth_path), ' appended. ')
            
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))

    
    # save image array as a pickle
    pickle_counter += 1
    pickle_filename = 'pickle_' + str(pickle_counter) + '.pkl'
    with open(pickle_filename, 'wb') as f:
        pkl.dump(image_array, f)
        print("Dumped one pickle")


# List all file paths of all files with .pgm extension in a list recursively
# depth_image_paths = glob.glob('/media/mihir/MKSSD/misc_offices_playroom_reception_studies_study_rooms/**/*.pgm', recursive=True)
# depth_image_paths = glob.glob('/media/mihir/MKSSD/cafe_dining_furniture_home_off_kitchen_libraries_living_rooms/**/*.pgm', recursive=True)
# depth_image_paths = glob.glob('/media/mihir/MKSSD/basements_bedrooms_bathrooms_bookstores/**/*.pgm', recursive=True)
depth_image_paths = glob.glob('D:/misc_offices_playroom_reception_studies_study_rooms/**/*.pgm', recursive=True)
#depth_image_path = glob.glob('D:/misc_offices_playroom_reception_studies_study_rooms/misc_part1/computer_lab_0001/*.pgm', recursive=True)
#depth_image_path = glob.glob('D:/cafe_dining_furniture_home_off_kitchen_libraries_living_rooms/furniture_stores/furniture_store_0001a/*.pgm', recursive=True)
#depth_image_path = glob.glob('D:/cafe_dining_furniture_home_off_kitchen_libraries_living_rooms/living_rooms_part2/living_room_0031/*.pgm', recursive=True)

# Save image paths to file
tenth_of_depth_images = depth_image_paths[::10]
#tenth_of_depth_images = depth_image_path[::300]
#tenth_of_depth_images = depth_image_path[:2]


with open('image_file_paths.txt', 'w') as file:
    for file_path in tenth_of_depth_images:
        file.write(file_path + '\n')


size_per_pickle = 1000
list_of_lists = [tenth_of_depth_images[i:i + size_per_pickle] for i in range(0, len(tenth_of_depth_images), size_per_pickle)]

print("Number of lists: {}".format(len(list_of_lists)))

print('Number of depth images: {}'.format(len(depth_image_paths)))
print('Number of depth images to process: {}'.format(len(tenth_of_depth_images)))

list_of_images = []
with open('image_file_paths.txt', 'r') as file:
    for file_path in file:
        list_of_images.append(file_path.rstrip('\n'))

for i in range(len(list_of_lists)):
    process_images(list_of_lists[i])


image_processes = []
with (open("pickle_1.pkl", "rb")) as imageProcessFile:
    while True:
        try:
            image_processes.append(pkl.load(imageProcessFile))
        except EOFError:
            break





def normalize_matrix(matrix):
    # find the minimum and maximum values in the matrix
    min_val = min(matrix.flatten())
    max_val = max(matrix.flatten())

    # compute the range of the values in the matrix
    val_range = max_val - min_val

    # create a new matrix with the same shape as the original
    normalized_matrix = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

    # normalize the values in the matrix by subtracting the minimum value and
    # dividing by the range
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            normalized_matrix[i][j] = (matrix[i][j] - min_val) / val_range
    return normalized_matrix

# Plotting
plotting = None
if plotting is not None:
    image_processes = image_processes[0]
    image_process_1 = image_processes[0]
    projected_depths = image_process_1.projected_depths
    #projected_depths_inv = normalize_matrix(projected_depths)
    completed_depths = image_process_1.completed_depths
    
    completed_depths_2 = image_process_1.completed_depths_2
    #completed_depths = np.clip(completed_depths_inv, 0.1, MAX_DEPTH)
    filled_depths = image_process_1.filled_depths
    #completed_depths_2[completed_depths_2 == 0] = MAX_DEPTH


    if plotting == 1:
        fig = plt.figure(figsize=(15, 7))
        fig.add_subplot(1, 3, 1)
        plt.imshow(projected_depths, cmap='gray', vmin = 0, vmax = MAX_DEPTH)
        fig.add_subplot(1, 3, 2)
        plt.imshow(completed_depths, cmap ='gray', vmin = 0, vmax = MAX_DEPTH)
        fig.add_subplot(1, 3, 3)
        plt.imshow(filled_depths, cmap ='gray', vmin = 0, vmax = MAX_DEPTH)
        # Display the plot
        plt.show()

    if plotting == 2:
        fig = plt.figure(figsize=(15, 7))
        fig.add_subplot(2, 2, 1)
        plt.imshow(projected_depths, cmap='gray', vmin = 0, vmax = MAX_DEPTH)
        fig.add_subplot(2, 2, 2)
        plt.imshow(completed_depths, cmap ='gray', vmin = 0, vmax = MAX_DEPTH)
        fig.add_subplot(2, 2, 3)
        plt.imshow(completed_depths_2, cmap ='gray', vmin = 0, vmax = MAX_DEPTH)
        fig.add_subplot(2, 2, 4)
        plt.imshow(filled_depths, cmap ='gray', vmin = 0, vmax = MAX_DEPTH)
        # Display the plot
        plt.show()
