
import os
from tqdm import tqdm
# images filedir: 
generated_images_dir = "/media/brianw/1511bdc1-b782-4302-9f3e-f6d90b91f857/home/brianw/ICRA_DATA/carla_data"
generated_images_dir = "/media/brianw/Elements/icra_data/training"


# Keep track of the number of images for each CE
ce_dict = {}
total_images = 0

# List every directory
all_experiments = os.listdir(generated_images_dir)

for exp_dir in tqdm(all_experiments):
    # Get the directory
    parent_folder = os.path.join(generated_images_dir, exp_dir)

    # Get current ce
    current_ce = "ce"+exp_dir[3]
    if current_ce not in ce_dict.keys():
        ce_dict[current_ce] = [0,0]  # Total number of images, number of experiments
    
    # Open the parent folder, and get the camera rgb files
    cam_folders = os.listdir(parent_folder)
    cam_folders = [x for x in cam_folders if "out_rgb" in x]
    for cam_folder in cam_folders:
        cam_images_dir = os.path.join(parent_folder, cam_folder)
        # Number of images
        num_images = len(os.listdir(cam_images_dir))

        ce_dict[current_ce][0] += num_images
        total_images += num_images

    ce_dict[current_ce][1] += 1

print(ce_dict)
print(total_images)

