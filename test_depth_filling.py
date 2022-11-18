import glob
import os

import numpy as np

from ip_basic.depth_map_utils import *
from ip_basic.vis_utils import *

from PIL import Image


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

def process_images(dataset_path, folders, fill_type, extrapolate, blur_type, 
    custom_kernel, morph_kernel, dilation_kernel, 
    small_hole_kernel, hole_kernel, large_hole_kernel):
    
    # Dataset path
    input_depth_dir = os.path.expanduser(dataset_path + folders)

    # Create output folder
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    outputs_dir = this_file_path + '/test_outputs'
    os.makedirs(outputs_dir, exist_ok=True)

    output_list = sorted(os.listdir(outputs_dir))
    output_folder_prefix = 'test_' 
    
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

    if not os.path.exists(output_depth_dir):
        os.makedirs(output_depth_dir)
    else:
        raise FileExistsError('Already exists!')

    
    # Save txt file with settings used
    with open(output_depth_dir + "/" + "settings" + str(last_output_index + 1) + ".txt", mode = "w+") as f:
        f.write("Settings for run " + str(last_output_index + 1) + ":")
        f.write("\n Fill type: " + fill_type)
        f.write("\n Blur type: " + str(blur_type))
        f.write("\n Extrapolate: " + str(extrapolate))
        
        if (fill_type == 'fast'):
            f.write("\n Custom Kernel type: " + str([name for name in globals() if globals()[name] is custom_kernel]))
            f.write("\n Morphology Kernel type: " + str([name for name in globals() if globals()[name] is morph_kernel]))
            f.write("\n Dilation Kernel type: " + str([name for name in globals() if globals()[name] is dilation_kernel]))
        
        if (fill_type == 'multiscale'):
            f.write("\n Small hole fill kernel type: " + str([name for name in globals() if globals()[name] is small_hole_kernel]))
            f.write("\n Hole fill kernel type: " + str([name for name in globals() if globals()[name] is hole_kernel]))
            f.write("\n Large hole fill Kernel type: " + str([name for name in globals() if globals()[name] is large_hole_kernel]))                
    
    folder = os.path.basename(os.path.normpath(input_depth_dir))
    
    # Get images in sorted order
    images = sorted(glob.glob(input_depth_dir + '/*.pgm'))

    # Use only every 10th image
    #images_to_use = images[::10]
    
    # Use only first image
    images_to_use = images[0]

    num_images = 1 #len(images_to_use)
   
    for i in range(num_images):

        depth_image_path = images_to_use#[i]

        depth_image = read_pgm(depth_image_path)
        projected_depths = convert_to_metres(depth_image)
        projected_depths = np.clip(projected_depths, 0, MAX_DEPTH)

        projected_depths[projected_depths == MAX_DEPTH] = MAX_DEPTH - 0.03
        projected_depths = (projected_depths).astype(np.float32)

        proj_depth_image_file_name = os.path.split(depth_image_path)[1]
        proj_file_path = output_depth_dir + "/" + folder + "_proj_" + proj_depth_image_file_name
        proj_depth_image = Image.fromarray((projected_depths/MAX_DEPTH*256).astype(np.uint8))    
        proj_depth_image.save(proj_file_path + ".png")

        # Fill in
        if fill_type == 'fast':
            final_depths = fill_in_fast(
                projected_depths, max_depth=MAX_DEPTH, extrapolate=extrapolate, blur_type=blur_type, 
                morph_kernel=morph_kernel, dilation_kernel=dilation_kernel)
        elif fill_type == 'multiscale':
            final_depths, process_dict = fill_in_multiscale(
                projected_depths, max_depth=MAX_DEPTH, extrapolate=extrapolate, blur_type=blur_type,
                show_process=False, small_hole_kernel=small_hole_kernel, hole_kernel=hole_kernel, large_hole_kernel=large_hole_kernel)
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))

        # Save depth map to a uint16 png (same format as disparity maps)
        depth_image_file_name = os.path.split(depth_image_path)[1]
        file_path = output_depth_dir + "/" + folder + "_final_" + depth_image_file_name
        depth_image = Image.fromarray((final_depths/MAX_DEPTH*256).astype(np.uint8))    
        depth_image.save(file_path + ".png")


def main():
    """Depth maps are saved to the 'outputs' folder.
    """

    folder11 = '/misc_part1/computer_lab_0001'

    folder21 = '/kitchens_part2/kitchen_0025'

    folder22 = '/living_rooms_part3/living_room_0045'
    
    datasetPath1 = 'D:/misc_offices_playroom_reception_studies_study_rooms' #+ folders1 #+ subFolder1
    datasetPath2 = 'D:/cafe_dining_furniture_home_off_kitchen_libraries_living_rooms' #+ folders2 #+ subFolder2
    datasetPath3 = 'D:/basements_bedrooms_bathrooms_bookstores' #+ folders3 #+ subFolder3

    # Best so far: 
    fill_type = 'fast'
    extrapolate = True
    blur_type = None
    
    # Fast fill kernels
    custom_kernel = DIAMOND_KERNEL_7
    morph_kernel = FULL_KERNEL_5
    dilation_kernel = FULL_KERNEL_7
    
    # Multiscale fill kernels
    small_hole_kernel=FULL_KERNEL_5
    hole_kernel=FULL_KERNEL_9
    large_hole_kernel=FULL_KERNEL_5
    
    # Processing
    process_images(datasetPath1, folder11, fill_type, extrapolate, blur_type, custom_kernel, 
        morph_kernel, dilation_kernel, small_hole_kernel, hole_kernel, large_hole_kernel)
    process_images(datasetPath2, folder21, fill_type, extrapolate, blur_type, custom_kernel, 
        morph_kernel, dilation_kernel, small_hole_kernel, hole_kernel, large_hole_kernel)
    process_images(datasetPath2, folder22, fill_type, extrapolate, blur_type, custom_kernel, 
        morph_kernel, dilation_kernel, small_hole_kernel, hole_kernel, large_hole_kernel)



if __name__ == "__main__":
    main()
