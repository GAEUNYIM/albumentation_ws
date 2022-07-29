from random import uniform
from random import seed
import math

seed(1)
for _ in range(10):
    code = math.floor(uniform(0.0, 3.0))

    if code == 0:
        w, h = (16, 9)
    elif code == 1:
        w, h = (4, 3)
    elif code == 2:
        w, h = (1, 1)


    real_height = 320 / w * h
    crop_height = (320 - real_height) / 2

    print(code, real_height, crop_height)