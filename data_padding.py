import cv2
import numpy as np
import os


path_source = "../../Media/flightData/13th/Tip"
path_dest = "../../Media/Tip" # TODO : You should create the directory before running python
os.makedirs(path_dest)

list_images = os.listdir(path_source)

src_x, src_y = 320, 240
base_x, base_y = 320, 320

start_x = (base_x - src_x) // 2
start_y = (base_y - src_y) // 2


base_img = np.full((base_y, base_x,  3), 0, dtype=np.uint8)

for file_name in list_images:
    # Read image
    img = cv2.imread(path_source + "/" + file_name)
    
    # Resize image
    img = cv2.resize(img, (src_x, src_y), interpolation=3)

    base_img[start_y:start_y + src_y, start_x:start_x + src_x] = img

    new_file_name = file_name.replace(".png", ".jpg")
    cv2.imwrite(path_dest + "/" + new_file_name, base_img)