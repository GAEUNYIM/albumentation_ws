import albumentations as A
import cv2
import os
import json


# Step / Which type of augmentation do you want to apply? (Important! Will be repeatedly used below) 
name_augmentation = "img_300x169" # TODO;



# Step / Set paths of image directories.
path_source_images = "../../Media/valid/img_4000x2250"
path_dest_images = "../../Media/valid/" + name_augmentation



# Returns lists of the file names
list_images = os.listdir(path_source_images)



# Construct an augmentation pipeline constructed
transform = A.Compose([ # TODO;
    A.LongestMaxSize(max_size=300, always_apply=True),
    ],
    bbox_params = A.BboxParams(format='coco', min_visibility=0, label_fields=['category_ids']),
)



# Step / Augment and Store new images, massks, annotations into a new directory
for file_name in list_images:

    # Read original data before augmentation
    image = cv2.imread(path_source_images + '/' + file_name)

    # Create new objects that are augmented
    augmentations = transform(image=image, bboxes=[], category_ids=[])
    augmentation_img = augmentations["image"]

    # Write new data after augmentation
    cv2.imwrite(path_dest_images + "/" + file_name, augmentation_img) # TODO;