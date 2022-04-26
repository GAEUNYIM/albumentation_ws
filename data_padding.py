import cv2
import numpy as np
import os


path_source = "../../Media/v4/valid/img_300x169"
path_dest = "../../Media/v4/valid/img_300x300_zero_padded" # TODO : You should create the directory before running python


list_images = os.listdir(path_source)


src_x, src_y = 300, 169
base_x, base_y = 300, 300

start_x = (base_x - src_x) // 2
start_y = (base_y - src_y) // 2
print(start_x, start_y)


base_img = np.full((base_y, base_x,  3), 0, dtype=np.uint8)

for file_name in list_images:
    img = cv2.imread(path_source + "/" + file_name)
    # print(np.shape(base_img[start_y:start_y + src_y, start_x:start_x+src_x]))
    base_img[start_y:start_y + src_y, start_x:start_x + src_x] = img

    new_file_name = file_name.replace(".JPG", ".jpg")
    cv2.imwrite(path_dest + "/" + new_file_name, base_img)