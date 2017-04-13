from __future__ import print_function
import glob

# image_folder = '/home/gao/Desktop/desk_image_and_pose'
image_folder = '/home/gao/Desktop/micro_switch_image_and_pose'
image_path_list = glob.glob(image_folder+'/*.png')
image_path_list = sorted(image_path_list)
image_path_list_file_path = image_folder+'/images.txt'
with open(image_path_list_file_path, 'w') as f:
    for image_path in image_path_list:
        print(image_path, file=f)
