import os
import glob

import numpy as np
import random
import PIL as im

from collections import OrderedDict
from matplotlib import pyplot as plt

# Create sample image array
#image_array = np.zeros([10, 10])

image_array = np.random.rand(10, 10)

image_array[0:2, :]     = 1
image_array[6:8, 3:5]   = 1
image_array[3:5, 6:8]   = 1
image_array[2:4, -2]    = 1
image_array[-1, -1]     = 1


# Image dimensions
N_ROWS = len(image_array[0])
N_COLS = len(image_array[1])
EMPTY = 1

# Find all empty pixels and the first index of each area
def find_empties(image):

    empties = []
    indices = (image == EMPTY).nonzero()

    for i in range(len(indices[0])):
        empties.append(indices[i][0], indices[i][1])
     
    return empties

# Finds the empty area connected to pixel [row, col] 
def find_empty_area(image_array, empties, row, col):
    # Add first empty pixel to emties and to queue
    empties.append([row, col])
    queue = [[row, col]]
    
    # Continue until no more new empty neighbor pixels in queue
    while len(queue) > 0:
        # Indexes of first pixel in queue
        r = queue[0][0]
        c = queue[0][1]
        
        # East neighbor
        if c < N_COLS - 1:
            if ([r, c + 1] not in empties) and ([r, c + 1] not in queue) and (image_array[r, c + 1] == 1):
                empties.append([r, c + 1])
                queue.append([r, c + 1])
        # South neighbor
        if r < N_ROWS - 1:
            if ([r + 1, c] not in empties) and ([r + 1, c] not in queue) and (image_array[r + 1, c] == 1):
                empties.append([r + 1, c])
                queue.append([r + 1, c])
        
        # For all rows except the top one    
        if r > row:
            # West neighbor
            if c > 0:
                if ([r, c - 1] not in empties) and ([r, c - 1] not in queue) and (image_array[r, c - 1] == 1):
                    empties.append([r, c - 1])
                    queue.append([r, c - 1])
            # North neighbor
            if ([r - 1, c] not in empties) and ([r - 1, c] not in queue) and (image_array[r - 1, c] == 1):
                empties.append([r - 1, c])
                queue.append([r - 1, c])
        
        # Remove finished pixel
        queue.remove([r, c])

        # Remove duplicates                                                             WHY IS THIS NECESARRY??
        unique_queue = []
        [unique_queue.append(pixel) for pixel in queue if pixel not in unique_queue]
        
        #unique_queue = list(OrderedDict.fromkeys(tuple(queue)))
        queue = unique_queue
        #queue = sorted(set(queue), key=lambda x: queue.index(x))
        
    #unique_empties = []
    #[unique_empties.append(pixel) for pixel in empties if pixel not in unique_empties]
    #unique_empties = list(OrderedDict.fromkeys(tuple(empties)))
    #empties = unique_empties
    #empties = sorted(set(empties), key=lambda x: empties.index(x))
    return empties


empties, empty_area_indexes = find_empties(image_array) 

print('Empties: ', empties)
print('Indexes: ', empty_area_indexes)

def find_nearest_valid(image_array, row, col):
    
    # Find distances to closest valid pixel in all directions
    east = N_COLS
    south = N_ROWS
    west = N_COLS
    north = N_ROWS
    
    for e in range(col, N_COLS):
        if image_array[row, e] < 1:
            east = e - col
            print('e ', e, 'east ', east)
            break

    for s in range(row, N_ROWS):
        if image_array[s, col] < 1:
            south = s - row
            print('s ', s, 'south ', south)
            break
    
    for w in range(col, -1, -1):
        if image_array[row, w] < 1:
            west = col - w
            print('w ', w, 'west ', west)
            break
    
    for n in range(row, -1, -1):
        if image_array[n, col] < 1:
            north = row - n
            print('n ', n, 'north ', north)
            break

    valid_neighbor_dist = [east, south, west, north]

    near_idx = valid_neighbor_dist.index(min(valid_neighbor_dist))
    #nearest_dir = valid_neighbor_dist[near_idx]
    print(valid_neighbor_dist)
    print(near_idx)
    if near_idx == 0:
        nearest_valid_pixel = [row, col + east]
    if near_idx == 1:
        nearest_valid_pixel = [row + south, col]
    if near_idx == 2:
        nearest_valid_pixel = [row, col - west]
    if near_idx == 3:
        nearest_valid_pixel = [row - north, col]    
    
    return nearest_valid_pixel

def fill_empty_spaces(image_array, empties):
    filled_image_array = image_array.copy()
    for r in range(0, N_ROWS):
        for c in range(0, N_COLS):
            if [r, c] in empties:
                nearest = find_nearest_valid(image_array, r, c)
                filled_image_array[r][c] = image_array[nearest[0], nearest[1]]
    return filled_image_array

filled_image_array = fill_empty_spaces(image_array, empties)
print('Image Array:', '\n', image_array)
print('Filled image Array:', '\n', filled_image_array)

fig = plt.figure(figsize=(10, 7))
fig.add_subplot(1, 2, 1)
plt.imshow(image_array, cmap='gray')
fig.add_subplot(1, 2, 2)
plt.imshow(filled_image_array, cmap ='gray')
# Display the plot
plt.show()



# Unused, not properly working
def fill_empty_areas(image_array, empties, empty_idxs):
    
    filled_image_array = image_array.copy()

    # Fill each area
    for area in range(len(empty_idxs)):
        # Start at first pixel in current area
        start_idx = empties.index(empty_idxs[area])
        print('area, start index', area, start_idx)

        # If last area
        if area == len(empty_idxs) - 1:
            # Stop index is last 
            stop_idx = -1
        else:
            # Stop index is pixel before first pixel in next area    
            stop_idx = empties.index(empty_idxs[area + 1])
        print('area, stop',area, stop_idx)

        empty_area = empties[start_idx:stop_idx]
        print('empties', empty_area)

        n_empties = len(empty_area)
        print('area #, n_empties: ', area, n_empties)

        for i in range (n_empties):
            idx = random.randint(0, len(empty_area)-1)
            print('length',len(empty_area))
            print('idx=',idx)
            pixel = empty_area[idx]
            print('pixel', pixel)
            nearest = find_nearest_valid(image_array, pixel[0], pixel[1])
            print('nearest', nearest)
            nearest_value = image_array[nearest]
            filled_image_array[pixel] = nearest_value
            empty_area.remove(pixel)

    return filled_image_array

#print(fill_empty_areas(image_array, empties, empty_area_indexes))


