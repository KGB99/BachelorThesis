import argparse
import os
from skimage import measure
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
import json
import math

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

    print("Loading annotation file...")
    f = open('Annotations/all_coco/all_coco.json')
    coco_dict = json.load(f)
    f.close()
    print("Loading annotation info file...")
    f = open('Annotations/all_coco/all_coco_info.json')
    info_dict = json.load(f)
    f.close()

    train_ratio = 0.7
    val_ratio = 0.15
    
    #iterate through bitmasks, calculate annotation and add to dictionary
    print('Creating annotations...')
    for camera in coco_dict:
        #indexes for the training and val cutoffs
        train_max_index = info_dict[camera] * (train_ratio)
        val_max_index = info_dict[camera] * (train_ratio + val_ratio)

        for i,img_id in enumerate(coco_dict[camera]):
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
    print('Annotaions done!')

    #write dictionaries to files
    print("Writing annotation files...")
    f = open("Annotations/train_" + output_name + ".json", "w")
    f.write(json.dumps(train_dict))
    f.close()
    f = open("Annotations/val_" + output_name + ".json", "w")
    f.write(json.dumps(val_dict))
    f.close()
    f = open("Annotations/test_" + output_name + ".json", "w")
    f.write(json.dumps(test_dict))
    f.close()

    print('OK')
