import albumentations as A
import cv2
import os
import json
from PIL import Image

from cv2 import IMWRITE_JPEG_QUALITY
from numpy import float32, int16, int32, uint16


# Step / Which type of augmentation do you want to apply? (Important! Will be repeatedly used below) 
name_augmentation = "img_test" # TODO;



# Step / Set paths of image directories.
path_source_images = "../../Media/valid/padding/img_4000x2250"
path_dest_images = "../../Media/valid/padding/" + name_augmentation



# Returns lists of the file names
list_images = os.listdir(path_source_images)



# Construct an augmentation pipeline constructed
transform = A.Compose([ # TODO;
    A.LongestMaxSize(max_size=300, interpolation=3, always_apply=True),
    ],
    bbox_params = A.BboxParams(format='coco', min_visibility=0, label_fields=['category_ids']),
)


# Step / Augment and Store new images, massks, annotations into a new directory
for file_name in list_images:

    # if file_name == 'DJI_0673.JPG':

        # Read original data before augmentation
        source_img_path = path_source_images + '/' + file_name
            
        print("Original jpg file size : ", str(os.path.getsize(source_img_path)) + " Bytes")

        image = cv2.imread(source_img_path)
        # print(type(image[0][0][0]))

        # Create new objects that are augmented
        augmentations = transform(image=image, bboxes=[], category_ids=[])
        augmentation_img = augmentations["image"]

        # Write new data after augmentation
        
        dest_img_path = path_dest_images + '/' + file_name
        cv2.imwrite(dest_img_path, augmentation_img) # TODO;

        # print(type(augmentation_img[0][0][0]))

        print("Augmented jpg file size : ", str(os.path.getsize(dest_img_path)) + " Bytes")