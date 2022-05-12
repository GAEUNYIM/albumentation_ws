import json
import os
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


# Step / Set paths # TODO;
path_source = "../../Media/v0.3/valid/" 
path_source_images = path_source + "images"
path_source_annotations = path_source + "annotations"
path_source_masks = path_source + "masks"



# Step / Set file name # TODO;
ann_filename = 'coco_annotations.json'
with open(path_source_annotations + '/' + ann_filename, 'r') as jf:
    json_data = json.load(jf)
    
# Check : Print out all the keys in json_data (You can easily see the structure of json file)
dict_keys = json_data.keys()
print(dict_keys)

# Get dictionaries
js_dicts_images = json_data["images"]
js_dicts_annotations = json_data["annotations"]




# step / Returns lists of the file names
list_images = os.listdir(path_source_images)



# Step / Create inversed dictionaries with "image_id"
ids_dict_with_file_names = {}

for dict in js_dicts_images:
    key = dict['file_name']

    value = dict['id']
    ids_dict_with_file_names[key] = value


# Step / Load coco data
coco_data = COCO(path_source_annotations + '/' + ann_filename)



# Step / get annotation ids list of file image
for file_name in list_images:
    
    if file_name == '200121_114652_736.jpg':
        # "191228_105517_252.jpg": # Test with this comlex figure

        # Get image id from file_name
        image_id = ids_dict_with_file_names[file_name]
        
        # Get annotation ids from file_name
        ann_ids = coco_data.getAnnIds(imgIds=image_id)
        
        # Load annotation dictionaries from annotation ids
        anns = coco_data.loadAnns(ann_ids)
        for ann in anns:
            mask = np.max(np.stack([coco_data.annToMask(ann) * ann["category_id"] * 100 for ann in anns]), axis=0)
        
        mask_name = file_name.replace('.jpg', '.png')

        # cv2.imwrite(mask_name, mask)
        cv2.imwrite(mask_name, mask)
