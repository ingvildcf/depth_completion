import glob
import os
import sys
import time

import cv2
import numpy as np
import png

from ip_basic.depth_map_utils import *
from ip_basic.vis_utils import *

from PIL import Image
import time


MAX_DEPTH = 10.00

DEPTH_PARAM1 = 351.30
DEPTH_PARAM2 = 1092.50

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



def main():
    """Depth maps are saved to the 'outputs' folder.
    """

    datasetPath_1 = 'D:/misc_offices_playroom_reception_studies_study_rooms'
    datasetPath_2 = 'D:/cafe_dining_furniture_home_off_kitchen_libraries_living_rooms'
    datasetPath_3 = 'D:/basements_bedrooms_bathrooms_bookstores'
    folder = '/misc_part1'
    subFolder = '/computer_lab_0001'
    
    ##############################
    # Options
    ##############################
    # Validation set
    input_depth_dir = os.path.expanduser(datasetPath_1 + folder + subFolder)
    data_split = 'val'

    # Test set
    # input_depth_dir = os.path.expanduser(
    #     '~/Kitti/depth/depth_selection/test_depth_completion_anonymous/velodyne_raw')
    # data_split = 'test'

    # Fast fill with Gaussian blur @90Hz (paper result)
    #fill_type = 'fast'
    #extrapolate = True
    #blur_type = 'gaussian'

    # Fast Fill with bilateral blur, no extrapolation @87Hz (recommended)
    #fill_type = 'fast'
    #extrapolate = False
    #blur_type = 'bilateral'

    # Multi-scale dilations with extra noise removal, no extrapolation @ 30Hz
    #fill_type = 'multiscale'
    # extrapolate = False
    # blur_type = 'bilateral'

    # Best so far: 
    fill_type = 'fast'
    extrapolate = True
    blur_type = None

    # Save output to disk or show process
    save_output = True

    ##############################
    # Processing
    ##############################
    if save_output:
        # Save to Disk
        show_process = False
        save_depth_maps = True
    else:
        if fill_type == 'fast':
            raise ValueError('"fast" fill does not support show_process')

        # Show Process
        show_process = True
        save_depth_maps = False

    # Create output folder
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    outputs_dir = this_file_path + '/outputs' + folder + subFolder
    os.makedirs(outputs_dir, exist_ok=True)

    output_folder_prefix = 'depth_' + data_split
    output_list = sorted(os.listdir(outputs_dir))
    if len(output_list) > 0:
        split_folders = [folder for folder in output_list
                         if folder.startswith(output_folder_prefix)]
        if len(split_folders) > 0:
            last_output_folder = split_folders[-1]
            last_output_index = int(last_output_folder.split('_')[-1])
        else:
            last_output_index = -1
    else:
        last_output_index = -1
    output_depth_dir = outputs_dir + '/{}_{:03d}'.format(
        output_folder_prefix, last_output_index + 1)

    if save_output:
        if not os.path.exists(output_depth_dir):
            os.makedirs(output_depth_dir)
        else:
            raise FileExistsError('Already exists!')
        #print('Output dir:', output_depth_dir)

    # Get images in sorted order
    images = sorted(glob.glob(input_depth_dir + '/*.pgm'))

    # Use only every 10th image
    images_to_use = images[::10]

    # Rolling average array of times for time estimation
    avg_time_arr_length = 10
    last_fill_times = np.repeat([1.0], avg_time_arr_length)
    last_total_times = np.repeat([1.0], avg_time_arr_length)

    num_images = len(images_to_use)
   
    for i in range(num_images):

        depth_image_path = images_to_use[i]

        # Calculate average time with last n fill times
        avg_fill_time = np.mean(last_fill_times)
        avg_total_time = np.mean(last_total_times)

        # Show progress
        sys.stdout.write('\rProcessing {} / {}, '
                         'Avg Fill Time: {:.5f}s, '
                         'Avg Total Time: {:.5f}s, '
                         'Est Time Remaining: {:.3f}s'.format(
                             i, num_images - 1, avg_fill_time, avg_total_time,
                             avg_total_time * (num_images - i)))
        sys.stdout.flush()

        # Start timing
        start_total_time = time.time()

        # Load depth projections from uint16 image
        #depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        #projected_depths = np.float32(depth_image / 256.0)

        depth_image = read_pgm(depth_image_path)
        projected_depths = convert_to_metres(depth_image)
        projected_depths = np.clip(projected_depths, 0, MAX_DEPTH)

        #Image.fromarray((projected_depths/10.0*256).astype(np.uint8)).show()
        #time.sleep(2)

        projected_depths[projected_depths == MAX_DEPTH] = MAX_DEPTH - 0.03
        projected_depths = (projected_depths).astype(np.float32)

        # Fill in
        start_fill_time = time.time()
        if fill_type == 'fast':
            final_depths = fill_in_fast(
                projected_depths, max_depth=10.0, extrapolate=extrapolate, blur_type=blur_type)
        elif fill_type == 'multiscale':
            final_depths, process_dict = fill_in_multiscale(
                projected_depths, max_depth=10.0, extrapolate=extrapolate, blur_type=blur_type,
                show_process=show_process)
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))
        end_fill_time = time.time()

        # Display images from process_dict
        if fill_type == 'multiscale' and show_process:
            
            #img_size = (570, 165) # Original

            img_size = (512, 424) # From 

            x_start = 80
            y_start = 50
            x_offset = img_size[0]
            y_offset = img_size[1]
            x_padding = 0
            y_padding = 28

            img_x = x_start
            img_y = y_start
            max_x = 1900

            row_idx = 0
            for key, value in process_dict.items():

                image_jet = cv2.applyColorMap(
                    np.uint8(value / np.amax(value) * 255),
                    cv2.COLORMAP_JET)
                cv2_show_image(
                    key, image_jet,
                    img_size, (img_x, img_y))

                img_x += x_offset + x_padding
                if (img_x + x_offset + x_padding) > max_x:
                    img_x = x_start
                    row_idx += 1
                img_y = y_start + row_idx * (y_offset + y_padding)

                # Save process images
                cv2.imwrite('process/' + key + '.png', image_jet)

            cv2.waitKey()

        # Save depth images to disk
        if save_depth_maps:
            depth_image_file_name = os.path.split(depth_image_path)[1]
            
            #Image.fromarray((final_depths/10.0*256).astype(np.uint8)).show()    
            #time.sleep(2)
            
            # Save depth map to a uint16 png (same format as disparity maps)
            file_path = output_depth_dir + '/' + depth_image_file_name #+ ".png"
            
            #depth_image = (final_depths/10.0 * 256).astype(np.uint8)
            d_im = Image.fromarray((final_depths/10.0*256).astype(np.uint8))    
            d_im.save(file_path + ".png")
            
            #with open(file_path, 'wb') as f:
                #depth_image = (final_depths * 256).astype(np.uint16)
                #depth_image = (final_depths/10.0 * 256).astype(np.uint8)
                #d_im = Image.fromarray((final_depths/10.0*256).astype(np.uint8))    
                #d_im.save(file_path)
                # pypng is used because cv2 cannot save uint16 format images
                #writer = png.Writer(width=depth_image.shape[1],
                #                    height=depth_image.shape[0],
                #                    bitdepth=16,
                #                    greyscale=True)
                #writer.write(f, depth_image)

        end_total_time = time.time()

        # Update fill times
        last_fill_times = np.roll(last_fill_times, -1)
        last_fill_times[-1] = end_fill_time - start_fill_time

        # Update total times
        last_total_times = np.roll(last_total_times, -1)
        last_total_times[-1] = end_total_time - start_total_time


if __name__ == "__main__":
    main()
