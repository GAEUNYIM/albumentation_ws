import albumentations as A
import cv2
import os
import json
from cv2 import BORDER_CONSTANT
from pycocotools.coco import COCO
import pycocotools.mask as pm
import numpy as np
from tqdm import tqdm
from itertools import groupby
import math
from random import uniform
from random import seed

'''
################################################
##### Description : What is this code for? #####
################################################

This code will help you to augment your data with padding images.
Additionally, you can also create a _new annotation file_ by modifying the extra parameters 
regarding the images that are affected by the padding (height, width, and box).
Here, we are supporting only bbox, but not also segmentation for the extra parameters.

########################################################
##### Detailed Guide : Please give attention here! #####
########################################################

If you are trying to excute this script for augmentation, please read this guide carefllly.
You may want to make sure the file paths by setting your customized paths.
If you have a strong understands of the following structure, then everything will go fine :)

1. Every data directory has 3 main subdirectories. 
Let's say you will augment '1_blade' data as source. 
Then, you might have the file structure below.

- 1_blade
------ annotations
----------- annotations_1_blade.json
------ images
----------- xxx{img_suffix}.jpg
----------- yyy{img_suffix}.jpg
----------- zzz{img_suffix}.jpg

2. By following the convention above, create a new target directory with appropriate name.
Let's say you will augment the sources by applying '2 main augmentation skills' below.

- rotation between (-15, 15) degrees
- resizing into 300 X 300

Then, please create following directories on your terminal. (Use '$ mkdir' commands)

- 1_blade_rotated_15_resized (This should be the appropriate name)
------ annotations
------ images

3. Finally, set our parameters below. 
They surely help you to augment your source data like a monkey magic! :)
'''


# Step / Which type of augmentation do you want to apply? (Important! Will be repeatedly used below) 
name_augmentation = "train" # Augmentation type # TODO;



# Path for images, and annotations 
path_source_images = "../../../Media/djistreaming/" # Path where source images located # TODO;
path_dest_images = "../../../Media/pp_djistreaming_4" # Path where augmented images located # TODO;

os.makedirs(path_dest_images)




# # Step / Open the original json annotations file
# old_ann_filename = 'coco_annotations.json' # TODO;
# with open(path_source_annotations + '/' + old_ann_filename, 'r') as jf:
#     json_data = json.load(jf)
# # Check : Print out all the keys in json_data (You can easily see the structure of json file)
# dict_keys = json_data.keys()
# print(dict_keys)
# # Split json : Dictionaries of images, and annotations (It is useful to handle each part of json file)
# js_dicts_images = json_data['images']
# js_dicts_annotations = json_data['annotations']
# print("number of original annotations : ", len(js_dicts_annotations))



# # Step / We want to create a new annotations file extracted from the original annotations file
# new_ann_filename = 'coco_annotations.json' # TODO;



# # Step / Prepare the new json file as a base structure, to modify it
# # Just copy it first
# js_dicts_new = json_data



# Returns lists of the file names
list_images = os.listdir(path_source_images)


# Choose a random number from 0 to 3
seed(1)

# Construct an augmentation pipeline constructed
height, width = 320, 320 # TODO;

transform_0 = A.Compose([ # TODO;
    A.Resize(height=180, width=320, interpolation=4),
    A.CropAndPad(px=(70, 0, 70, 0), pad_mode=BORDER_CONSTANT, pad_cval=0, keep_size=False, sample_independently=False, p=1.0)
    ],
    bbox_params = A.BboxParams(format='coco', min_visibility=0, label_fields=['category_ids']),
)

transform_1 = A.Compose([ # TODO;
    A.Resize(height=240, width=320, interpolation=4),
    A.CropAndPad(px=(40, 0, 40, 0), pad_mode=BORDER_CONSTANT, pad_cval=0, keep_size=False, sample_independently=False, p=1.0)
    ],
    bbox_params = A.BboxParams(format='coco', min_visibility=0, label_fields=['category_ids']),
)

transform_2 = A.Compose([ # TODO;
    A.Resize(height=320, width=320, interpolation=4),
    ],
    bbox_params = A.BboxParams(format='coco', min_visibility=0, label_fields=['category_ids']),
)


cnt_0, cnt_1, cnt_2 = 0, 0, 0

# Step / Augment and Store new images, massks, annotations into a new directory
for file_name in tqdm(list_images):

    # Read original data before augmentation
    image = cv2.imread(path_source_images + '/' + file_name)

    # Randomly select one of 3 ratios
    code = math.floor(uniform(0.0, 3.0))
    if code == 0:
        cnt_0 += 1
    elif code == 1:
        cnt_1 += 1
    else:
        cnt_2 += 1


    if code == 0: # 16:9
        augmentations = transform_0(image=image, bboxes=[], mask=np.array([[]]), category_ids=[]) # WARN : Mask should be ndarray !!!
    elif code == 1: # 4:3
        augmentations = transform_1(image=image, bboxes=[], mask=np.array([[]]), category_ids=[])  # WARN : Mask should be ndarray !!!
    elif code == 2: # 1:1
        augmentations = transform_2(image=image, bboxes=[], mask=np.array([[]]), category_ids=[])  # WARN : Mask should be ndarray !!!

    augmentation_img = augmentations["image"] # BUG : Image is only one

    cv2.imwrite(path_dest_images + "/" + file_name, augmentation_img) # TODO;

