from pycocotools.coco import COCO
import numpy as np 
import pandas as pd
from tqdm import tqdm
import os
import json
import cv2

img_dir = '../../Media/v2/valid/img_300x300'
coco_path = '../../Media/v2/valid/annotations/empty_coco_annotations.json'

custom_coco_dict = {}

# set categories
cat_info = []

cat_names = ['blade', 'nose', 'pole']
label_ids = [1, 2, 3]

for label_id, cat_name in zip(label_ids, cat_names):
    cat_dict = {
               "supercategory": 'windturbine',
               "id": label_id,
               "name": cat_name
               }
    cat_info.append(cat_dict)
custom_coco_dict['categories'] = cat_info

img_info = []
annot_info = []
annot_id = 0


file_names = os.listdir(img_dir)
annot_id = 0 
    
for i, img_file_name in enumerate(file_names):
        
    img_id = i 
    img_path = os.path.join(img_dir, img_file_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_dict = {
        "file_name": img_file_name,
        "height": 300,
        "width": 300,
        "id": img_id
    }

    img_info.append(img_dict)
    bbox = [150, 150, 10, 10]
    annot_dict={
            "bbox": bbox,
            "category_id": 0,
            "id": annot_id,
            "iscrowd": 0,
            "area": bbox[2]*bbox[3],
            "image_id": img_id,
            "segmentation": []
        }
    annot_info.append(annot_dict)
    annot_id += 1 

custom_coco_dict["images"] = img_info
custom_coco_dict["annotations"] = annot_info
        

with open(coco_path, "w") as json_file:
    json.dump(custom_coco_dict, json_file)


