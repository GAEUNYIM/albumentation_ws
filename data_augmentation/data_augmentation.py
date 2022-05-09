import albumentations as A
import cv2
import os
import json
from cv2 import BORDER_CONSTANT
from pycocotools.coco import COCO
import pycocotools.mask as pm
import numpy as np
from itertools import groupby
# from skimage import measure

'''
################################################
##### Description : What is this code for? #####
################################################

This code will help you to augment your data with various transformations.
Augmentation of bbox based on segmentation is more accurate than based on bbox.
Here, you can apply any augmenetaion.

'''



# Step /
name_augmentation = "rotated2" # Augmentation type # TODO;

# Path for source images, and annotations, and masks
path_source = "../../../Media/v0/valid/per_case/blade/" # Path where source images located # TODO;
path_source_images = path_source + "images" 
path_source_annotations = path_source + "annotations"
path_source_masks = path_source + "masks" 

# Path for destination images, and annotations, and masks
path_dest = "../../../Media/v5/valid/" + name_augmentation + "/" # Path where augmented images located # TODO;
# os.makedirs(path_dest)

path_dest_images = path_dest + "images"
path_dest_annotations = path_dest + "annotations" 
path_dest_masks = path_dest + "masks" 
# os.makedirs(path_dest_images)
# os.makedirs(path_dest_annotations)
# os.makedirs(path_dest_masks)



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
width, height = 300, 300 # TODO;

transform = A.Compose([ # TODO;
    # A.Resize(width=300, height=300, interpolation=3),
    # A.CenterCrop(width=300,height=168, p=1),
    # A.CropAndPad(px=(66, 0, 66, 0), pad_mode=BORDER_CONSTANT, pad_cval=0, 
    #         keep_size=False, sample_independently=False, p=1.0)
    # A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=1),
    # A.HorizontalFlip(p=1)
    A.Rotate(limit=[15, 15], p=1)
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
masks_dict_with_key_id = {}

for dict in js_dicts_annotations:
    key = dict['image_id']
    bboxes_dict_with_key_id[key] = []
    category_ids_dict_with_key_id[key] = []
    masks_dict_with_key_id[key] = []

for dict in js_dicts_annotations:
    key = dict['image_id']

    value = dict['bbox']
    bboxes_dict_with_key_id[key].append(value)

    value = dict['category_id']
    category_ids_dict_with_key_id[key].append(value)

    value = dict['segmentation'][0]
    masks_dict_with_key_id[key].append(value)

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

# Step / Augment and Store new images, massks, annotations into a new directory
for file_name in list_images:

    # if file_name == '191228_105517_252.jpg': # TODO; Test with your own figure!

    image_id = ids_dict_with_file_names[file_name]

    # Read original data before augmentation
    image = cv2.imread(path_source_images + '/' + file_name)
    bboxes = bboxes_dict_with_key_id[image_id]
    category_ids = category_ids_dict_with_key_id[image_id]

    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids) 

    augmentation_zip = {} # Key : ann_id, Value : [cat_id, segmentation] 

    # Create new objects that are augmented
    for ann in anns:
        mask = coco.annToMask(ann)
        cat_id = ann["category_id"]
        ann_id = ann["id"]

        augmentations = transform(image=image, bboxes=bboxes, mask=mask, category_ids=category_ids) # WARN : Mask should be ndarray !!!
        augmentation_img = augmentations["image"] # BUG : Image is only one
        augmentation_mask = augmentations["mask"] # Array filled with 0, 1, 2, 3

        fortran_ground_truth_binary_mask = np.asfortranarray(augmentation_mask) # Problem
        encoded_ground_truth = pm.encode(fortran_ground_truth_binary_mask) # RLE Object?
        area = float(pm.area(encoded_ground_truth))
        bbox = list(pm.toBbox(encoded_ground_truth))

        segmentation = binary_mask_to_rle(fortran_ground_truth_binary_mask)

        augmentation_zip[ann_id] = [cat_id, area, bbox, segmentation]


    # Append new "images" dictionary info; The otherse are maintained inside
    img_dict = {
        "file_name": file_name,
        "height": 300, # TODO; 
        "width": 300, # TODO;
        "id": image_id
    }
    new_img_info.append(img_dict)
    img_id += 1

    annot_dicts = []

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
        annot_dicts.append(annot_dict)
        annot_id += 1 
        
    # Write new data after augmentation
    cv2.imwrite(path_dest_images + "/" + file_name, augmentation_img) # TODO;

    # TODO; Activate below if you want masks/ directory
    # mask_name = file_name.replace('.jpg', '.png') 
    # mask = np.max(np.stack([coco.annToMask(ann) * ann["category_id"] * 100 for ann in annot_dicts]), axis=0)
    # if file_name == '191228_105517_252.jpg': # TODO; Test with your own figure!
    #     cv2.imwrite("hi.png", mask) # For test;
    # cv2.imwrite(path_dest_masks + "/" + mask_name, augmentation_mask)


# print("annot_info[0] : ", annot_info[0])
js_dicts_new['images'] = new_img_info
js_dicts_new['annotations'] = new_annot_info

print("img_id : ", img_id - 1)
print("annot_id : ", annot_id - 1)

# Step / Write down a new josn annotations file # TODO;
with open(path_dest_annotations + "/" + new_ann_filename, 'w', encoding='utf-8') as ef:
    json.dump(js_dicts_new, ef, ensure_ascii=False, indent="\t")