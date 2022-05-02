import json
import os
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

path_source = "../../Media/v0/valid/"
path_source_annotations = path_source + "annotations"
path_source_masks = path_source + "masks"

ann_file_name = 'coco_annotations.json'
with open(path_source_annotations + '/' + ann_file_name, 'r') as jf:
    json_data = json.load(jf)
    
# Check : Print out all the keys in json_data (You can easily see the structure of json file)
dict_keys = json_data.keys()
print(dict_keys)

