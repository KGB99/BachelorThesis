#!/bin/bash
Path="/Users/kerim/dev/BachelorThesis"
python3 code/MakeCoco.py --path_image="$Path/Data_sample/mvpsp/train/001000/rgb" --path_bitmask="$Path/Data_sample/mvpsp/train/001000/mask"
