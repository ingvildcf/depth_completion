import numpy as np
import pickle as pkl
import sys
import random

import threading
import time
import os
import glob

import concurrent.futures
import threading
from threading import Thread, Lock


#from pickles import *

MAX_DEPTH = 7.50

DEPTH_PARAM1 = 351.30
DEPTH_PARAM2 = 1092.50


aggregated_array = []

IMAGES_PER_PICKLE = 1000

GLOBAL_PICKLE_COUNTER = 0

SAVE_PATH = "outputs2/"

#### Snippet to parse through the folder path and list out the .pgm files in a .txt file 

datasetPath_1 = 'D:\misc_offices_playroom_reception_studies_study_rooms' #~/misc_offices_playroom_reception_studies_study_rooms'
folder = '/misc_part1'
#subFolder = '/computer_lab_0001'
folder1 = 'misc1'

files = glob.glob(datasetPath_1 + folder + '/**/*.pgm', recursive=True)
files.sort()
print(len(files))

# save files to a .txt file
with open(SAVE_PATH + 'textfiles/' + folder1 + '.txt', 'w+') as f:
    for item in files:
        f.write("%s\n" % item)




# Functions to read .pgm images and convert them to numpy arrays


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


def save_pickle():
    global aggregated_array
    global GLOBAL_PICKLE_COUNTER
    print("Saving pickle")
    # write numbered pickle file
    with open(SAVE_PATH + 'nyu_depth_{}.pickle'.format(GLOBAL_PICKLE_COUNTER), 'wb') as f:
        pkl.dump(aggregated_array, f)
        print("Saved pickle: {}".format(GLOBAL_PICKLE_COUNTER))
        GLOBAL_PICKLE_COUNTER += 1

def append_to_array(np_array):
    global aggregated_array
    aggregated_array.append(np_array)
    if len(aggregated_array) == IMAGES_PER_PICKLE/2:
        print("Aggregated array length is {}".format(len(aggregated_array)))
    if len(aggregated_array) == IMAGES_PER_PICKLE:
        save_pickle()
        aggregated_array = []


def read_file_list(filelist_path):
    with open(filelist_path, 'r') as f:
        filelist = f.readlines()
    return filelist


def save_images_from_text_file_to_pickle(filelist_path, percentage_onwards):
    global DEPTH_PARAM1
    global DEPTH_PARAM2
    global MAX_DEPTH

    if percentage_onwards > 100:
        print("Percentage onwards cannot be greater than 100")
        return

    filelist = read_file_list(filelist_path)

    file_ctr = 0.0
    len_filelist = float(len(filelist))

    filelist = filelist[int(len_filelist*percentage_onwards/100.0):]

    for file in filelist:
        file = file.strip()
        img = read_pgm(file)
        if img is None:
            continue
        depth = np.array(img, dtype=np.float32)
        depth = np.clip(DEPTH_PARAM1 /  (DEPTH_PARAM2 - depth), 0, MAX_DEPTH)
        append_to_array(depth)

        file_ctr += 1.0

        if file_ctr % 100 == 0:
            percent_complete = file_ctr * 100.0 / (len_filelist)
            discretization = int(percent_complete / 2.0)
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("[%-50s] %d%%" % ('='*discretization, 2*discretization))
            sys.stdout.flush()
    save_pickle()

## Multithreading the save pickle process. This function allows the saving of a file per thread

gpc_mutex = Lock()

def per_thread_save_pickles(list_of_files):
    global DEPTH_PARAM1
    global DEPTH_PARAM2
    global MAX_DEPTH
    global GLOBAL_PICKLE_COUNTER

    aggregated_array = []

    start_time = time.time()
    for file in list_of_files:
        file = file.strip()
        img = read_pgm(file)
        if img is None:
            continue
        depth = np.array(img, dtype=np.float32)
        depth = np.clip(DEPTH_PARAM1 /  (DEPTH_PARAM2 - depth), 0, MAX_DEPTH)
        aggregated_array.append(depth)
    print("finished reading at: {}".format(time.time()- start_time))

    # write numbered pickle file
    gpc_mutex.acquire()
    GLOBAL_PICKLE_COUNTER += 1
    counter_value = GLOBAL_PICKLE_COUNTER
    gpc_mutex.release()
    with open(SAVE_PATH + 'nyu_depth_{}.pickle'.format(counter_value), 'wb') as f:
        pkl.dump(aggregated_array, f)
        print("Saved pickle: {}".format(counter_value))
    print("Written pickle: {}".format(counter_value))
    print("Thread id: {} eixitng.".format(threading.get_ident()))



## This one tries to clean the generated pickle file in case of insconsistencies.
# parse folder to check pickle files


def get_file_list(path):
    files = os.listdir(path)
    files = [path + f for f in files if f.endswith('.pickle')]
    return files


# load pickle files and check length of aggregated array
def check_pickle_files(path):
    files = get_file_list(path)
    for file in files:
        array = []
        with open(file, 'rb') as f:
            array = pkl.load(f)
            if type(array[-1]) is not np.ndarray:
                print("Last element of file {} is not a numpy array".format(file))
                array[-1] = array[-1][0]


            if len(array) < 1000:
                print("File: {} has length: {}".format(file, len(array)))
                while len(array) < 1000:
                    array.append(random.sample(array, 1)[0])
                with open(file+"duplicate", 'wb') as f:
                    pkl.dump(array, f)
                    print("Saved duplicate file: {}".format(file+"duplicate"))


# Actually multithreading and saving the pickle files together. Run this to save multiple pickles

shortened_files = []
for i in range(0, len(files), 10):
    shortened_files.append(files[i])

list_of_lists = []

for i in range(0, len(shortened_files), 1000):
    list_of_lists.append(shortened_files[i:i+1000])

print(len(list_of_lists))

for filelist_names in list_of_lists:
    per_thread_save_pickles(filelist_names)
    with open(SAVE_PATH + "written_names.txt", "a") as f:
            f.writelines(filelist_names)

check_pickle_files(SAVE_PATH)

