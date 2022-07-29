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
name_augmentation = "valid" # Augmentation type # TODO;



# Path for images, and annotations 
path_source = "../../../Media/v0/" + name_augmentation # Path where source images located # TODO;
path_dest = "../../../Media/v9/" + name_augmentation # Path where augmented images located # TODO;
os.makedirs(path_dest)

path_source_images = path_source + "/images" 
path_source_annotations = path_source + "/annotations"

path_dest_images = path_dest + "/images"
path_dest_annotations = path_dest + "/annotations" 
os.makedirs(path_dest_images)
os.makedirs(path_dest_annotations)



# Step / Open the original json annotations file
old_ann_filename = 'coco_annotations.json' # TODO;
with open(path_source_annotations + '/' + old_ann_filename, 'r') as jf:
    json_data = json.load(jf)
# Check : Print out all the keys in json_data (You can easily see the structure of json file)
dict_keys = json_data.keys()
print(dict_keys)
# Split json : Dictionaries of images, and annotations (It is useful to handle each part of json file)
js_dicts_images = json_data['images']
js_dicts_annotations = json_data['annotations']
print("number of original annotations : ", len(js_dicts_annotations))



# Step / We want to create a new annotations file extracted from the original annotations file
new_ann_filename = 'coco_annotations.json' # TODO;



# Step / Prepare the new json file as a base structure, to modify it
# Just copy it first
js_dicts_new = json_data



# Returns lists of the file names
list_images = os.listdir(path_source_images)


# Choose a random number from 0 to 3
seed(1)

# Construct an augmentation pipeline constructed
height, width = 320, 320 # TODO;

transform_0 = A.Compose([ # TODO;
    A.Resize(height=320, width=320, interpolation=3),
    A.CenterCrop(height=180, width=320),
    A.CropAndPad(px=(70, 0, 70, 0), pad_mode=BORDER_CONSTANT, pad_cval=0, keep_size=False, sample_independently=False, p=1.0)
    ],
    bbox_params = A.BboxParams(format='coco', min_visibility=0, label_fields=['category_ids']),
)

transform_1 = A.Compose([ # TODO;
    A.Resize(height=320, width=320, interpolation=3),
    A.CenterCrop(height=240, width=320),
    A.CropAndPad(px=(40, 0, 40, 0), pad_mode=BORDER_CONSTANT, pad_cval=0, keep_size=False, sample_independently=False, p=1.0)
    ],
    bbox_params = A.BboxParams(format='coco', min_visibility=0, label_fields=['category_ids']),
)

transform_2 = A.Compose([ # TODO;
    A.Resize(height=320, width=320, interpolation=3),
    ],
    bbox_params = A.BboxParams(format='coco', min_visibility=0, label_fields=['category_ids']),
)

# Define img & annotation id
img_id = 1
annot_id = 1
new_img_info = []
new_annot_info = []

print("number of images : ", len(list_images))

# Step / Create inversed dictionaries with "image_id"
ids_dict_with_file_names = {}
bboxes_dict_with_key_id = {}
category_ids_dict_with_key_id = {}

for dict in js_dicts_annotations:
    key = dict['image_id']
    bboxes_dict_with_key_id[key] = []
    category_ids_dict_with_key_id[key] = []

for dict in js_dicts_annotations:
    key = dict['image_id']

    value = dict['bbox']
    bboxes_dict_with_key_id[key].append(value)

    value = dict['category_id']
    category_ids_dict_with_key_id[key].append(value)

for dict in js_dicts_images:
    key = dict['file_name']

    value = dict['id']
    ids_dict_with_file_names[key] = value



def visualize_image_with_bbox(img, bboxes, file_name):
    for bbox in bboxes:
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[0] + bbox[2])
        y_max = int(bbox[1] + bbox[3])
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 256, 0), 1)
    cv2.imwrite(path_dest + file_name, img)

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

coco = COCO(path_source_annotations + "/" + old_ann_filename)

cnt_0, cnt_1, cnt_2 = 0, 0, 0

# Step / Augment and Store new images, massks, annotations into a new directory
for file_name in tqdm(list_images):

    # if file_name == '191228_105517_252.jpg': # TODO; Test with your own figure!

    image_id = ids_dict_with_file_names[file_name]

    # Read original data before augmentation
    image = cv2.imread(path_source_images + '/' + file_name)

    bboxes = bboxes_dict_with_key_id[image_id]
    category_ids = category_ids_dict_with_key_id[image_id]

    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids) 

    augmentation_zip = {} # Key : ann_id, Value : [cat_id, segmentation] 

    code = math.floor(uniform(0.0, 3.0))
    if code == 0:
        cnt_0 += 1
    elif code == 1:
        cnt_1 += 1
    else:
        cnt_2 += 1

    # Create new objects that are augmented
    for ann in anns:
        mask = coco.annToMask(ann)
        cat_id = ann["category_id"]
        ann_id = ann["id"]

        if code == 0: # 16:9
            augmentations = transform_0(image=image, bboxes=bboxes, mask=mask, category_ids=category_ids) # WARN : Mask should be ndarray !!!
        elif code == 1: # 4:3
            augmentations = transform_1(image=image, bboxes=bboxes, mask=mask, category_ids=category_ids) # WARN : Mask should be ndarray !!!
        elif code == 2: # 1:1
            augmentations = transform_2(image=image, bboxes=bboxes, mask=mask, category_ids=category_ids) # WARN : Mask should be ndarray !!!

        augmentation_img = augmentations["image"] # BUG : Image is only one
        augmentation_mask = augmentations["mask"] # Array filled with 0, 1, 2, 3

        fortran_ground_truth_binary_mask = np.asfortranarray(augmentation_mask) # Problem
        encoded_ground_truth = pm.encode(fortran_ground_truth_binary_mask) # RLE Object?
        area = float(pm.area(encoded_ground_truth))
        bbox = list(pm.toBbox(encoded_ground_truth))

        segmentation = binary_mask_to_rle(fortran_ground_truth_binary_mask)
        # print(segmentation)

        augmentation_zip[ann_id] = [cat_id, area, bbox, segmentation]


    # Append new "images" dictionary info; The otherse are maintained inside
    img_dict = {
        "file_name": file_name,
        "height": 320,
        "width": 320,
        "id": image_id
    }
    new_img_info.append(img_dict)
    img_id += 1

    # Append new "annotations" dictionary info; The otherse are maintained inside
    for ann in anns:

        ann_id = ann["id"]
        aug_zip = augmentation_zip[ann_id]
        # contour = np.flip(contour, axis=1)
        # segmentation = contour.ravel().tolist()

        annot_dict = {
            "bbox" : aug_zip[2], # Changed;
            "category_id": aug_zip[0],
            "id": annot_id, # Increased;
            "iscrowd": 0, # Fixed;
            "area": aug_zip[1], #js_dicts_annotations[i]['area'],
            "image_id": image_id,
            "segmentation": [aug_zip[3]]
        }
        new_annot_info.append(annot_dict)
        annot_id += 1 

    # Write new data after augmentation
    cv2.imwrite(path_dest_images + "/" + file_name, augmentation_img) # TODO;


# print("annot_info[0] : ", annot_info[0])
js_dicts_new['images'] = new_img_info
js_dicts_new['annotations'] = new_annot_info

print("img_id : ", img_id - 1)
print("annot_id : ", annot_id - 1)

# Step / Write down a new josn annotations file # TODO;
with open(path_dest_annotations + "/" + new_ann_filename, 'w', encoding='utf-8') as ef:
    json.dump(js_dicts_new, ef, ensure_ascii=False, indent="\t")

print("stats : ", cnt_0, cnt_1, cnt_2)
print("stats : ", cnt_0/img_id, "%, ", cnt_1/img_id, "%, ", cnt_2/img_id, "%, ")