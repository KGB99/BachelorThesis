import argparse
import os
from skimage import measure
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
import json
import math

train_max_index = 50
val_max_index = 70

if __name__ == "__main__":
    #np.set_printoptions(threshold=np.inf)
    # parse arguments from command line
    parser = argparse.ArgumentParser(description="This program transforms images and their bit masks to COCO dataset formatted segmentation annotations.")
    parser.add_argument("--output_file", help="Name for the output file.", required=True, type=str)
    parser.add_argument("--amodal", help="Write true if you want to calculate the amodal masks, default is modal", required=False, type=bool, default=False)
    args = parser.parse_args()
    output_name = args.output_file
    AMODAL = args.amodal

    if not os.path.isdir('Annotations'):
        os.mkdir('Annotations')

    #create dictionary for the annotations in COCO style
    train_dict = {} 
    train_dict["annotations"] = []
    train_dict["info"] = {"description" : "COCO dataset annotations for the medical dataset from cvg's Jonas Hein"}
    train_dict["licenses"] = {}
    train_dict["images"] = []

    val_dict = {} 
    val_dict["annotations"] = []
    val_dict["info"] = {"description" : "COCO dataset annotations for the medical dataset from cvg's Jonas Hein"}
    val_dict["licenses"] = {}
    val_dict["images"] = []

    test_dict = {} 
    test_dict["annotations"] = []
    test_dict["info"] = {"description" : "COCO dataset annotations for the medical dataset from cvg's Jonas Hein"}
    test_dict["licenses"] = {}
    test_dict["images"] = []

    f = open('Annotations/all_coco.json')
    coco_dict = json.load(f)
    f.close()

    #iterate through bitmasks, calculate annotation and add to dictionary
    print('Creating annotations...')
    i = 0
    for camera in coco_dict:
        for img_id in coco_dict[camera]:
            img_dict = coco_dict[camera][img_id]
            if (i > val_max_index):
                test_dict["annotations"].append(img_dict['mask'])
                test_dict["images"].append(img_dict['img'])
            elif (i > train_max_index):
                val_dict["annotations"].append(img_dict['mask'])
                val_dict["images"].append(img_dict['img'])
            else:
                train_dict["annotations"].append(img_dict['mask'])
                train_dict["images"].append(img_dict['img']) 
            i = i + 1
    print('Annotaions done!')

    #write dictionaries to files
    f = open("Annotations/train_all_coco.json", "w")
    f.write(json.dumps(train_dict, indent=3))
    f.close()
    f = open("Annotations/val_all_coco.json", "w")
    f.write(json.dumps(val_dict, indent=3))
    f.close()
    f = open("Annotations/test_all_coco.json", "w")
    f.write(json.dumps(test_dict, indent=3))
    f.close()

    print('OK')
