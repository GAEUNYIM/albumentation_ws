import albumentations as A
import cv2
import os
import json
# import numpy as np
# from collections import OrderedDict
# import matplotlib.pyplot as plt
# from PIL import Image

'''
################################################
##### Description : What is this code for? #####
################################################

This code will help you to augment your data with rotating images.
Additionally, you can also create a _new annotation file_ by modifying the extra parameters 
regarding the images that are affected by the rotation (height, width, and box).
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
name_augmentation = "rotated" # Augmentation type # TODO;

# Path for images, and annotations 
path_source = "../../Media/valid/per_case/blade/" # Path where source images located # TODO;
path_dest = "../../Media/valid/per_case/" + name_augmentation + "/" # Path where augmented images located # TODO;

path_source_images = path_source + "images" 
path_source_annotations = path_source + "annotations"

path_dest_images = path_dest + "images"
path_dest_annotations = path_dest + "annotations" 



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


# Construct an augmentation pipeline constructed
# height, width = 300, 300 # TODO;

transform = A.Compose([ # TODO;
    # A.Resize(height=height, width=width),
    # A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=1),
    # A.HorizontalFlip(p=1)
    A.SafeRotate(limit=[10, 10], p=1)
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



# Step / Augment and Store new images, massks, annotations into a new directory
for file_name in list_images:

    image_id = ids_dict_with_file_names[file_name]

    # Read original data before augmentation
    image = cv2.imread(path_source_images + '/' + file_name)
    bboxes = bboxes_dict_with_key_id[image_id]
    category_ids = category_ids_dict_with_key_id[image_id]

    # Create new objects that are augmented
    augmentations = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    augmentation_img = augmentations["image"]
    augmentation_bboxes = augmentations["bboxes"]

    # Write new data after augmentation
    cv2.imwrite(path_dest_images + "/" + file_name, augmentation_img) # TODO;

    # Append new "images" dictionary info; The otherse are maintained inside
    img_dict = {
        "file_name": file_name,
        "height": 512,
        "width": 512,
        "id": image_id
    }
    new_img_info.append(img_dict)
    img_id += 1

    # Append new "annotations" dictionary info; The otherse are maintained inside
    for bbox, cat_id in zip(augmentation_bboxes, category_ids):
        annot_dict = {
            "bbox" : bbox, # Changed;
            "category_id": cat_id,
            "id": annot_id, # Increased;
            "iscrowd": 0, # Fixed;
            "area": 0, #js_dicts_annotations[i]['area'],
            "image_id": image_id,
            "segmentation": [], #js_dicts_annotations[i]['segmentation']
        }
        new_annot_info.append(annot_dict)
        annot_id += 1 

    if file_name == '191228_112311_507.jpg': # '191228_102857_995.jpg':
        print("original bboxes : ", bboxes)
        print("original categories : ", category_ids)
        visualize_image_with_bbox(image, bboxes, 'original.jpg')

        print("augmented bbox : ", augmentation_bboxes)
        visualize_image_with_bbox(augmentation_img, augmentation_bboxes, 'augmented.jpg')
    
    if len(bboxes) != len(augmentation_bboxes):
        print(file_name, " at id ", image_id)

# print("annot_info[0] : ", annot_info[0])
js_dicts_new['images'] = new_img_info
js_dicts_new['annotations'] = new_annot_info

print("img_id : ", img_id - 1)
print("annot_id : ", annot_id - 1)

# Step / Write down a new josn annotations file # TODO;
with open(path_dest_annotations + "/" + new_ann_filename, 'w', encoding='utf-8') as ef:
    json.dump(js_dicts_new, ef, ensure_ascii=False, indent="\t")