import argparse
import os
import json


if __name__ == "__main__":
    #np.set_printoptions(threshold=np.inf)
    # parse arguments from command line
    parser = argparse.ArgumentParser(description="This program transforms images and their bit masks to COCO dataset formatted segmentation annotations.")
    parser.add_argument("--output_dir", help="Name for the output dir.", required=True, type=str)
    parser.add_argument("--coco_file", help="the path to the coco file containing all infos", required=True, type=str)
    parser.add_argument("--train_ratio", required=False, type=float, default=0.9)
    #TODO:
    parser.add_argument("--test", help="type true if creating testing annotations so it doesnt create two files", required=False, default=False, type=bool)
    
    args = parser.parse_args()
    output_dir = args.output_dir
    coco_dir = args.coco_file
    train_ratio = args.train_ratio
    TEST = args.test

    if not os.path.isdir('./Annotations'):
        os.mkdir('./Annotations')

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
    # './Annotations/test_coco/test_coco.json'
    f = open(coco_dir)
    coco_dict = json.load(f)
    f.close()  
    print("Loading annotation info file...")
    f = open(coco_dir[::-1].split('.')[1][::-1] + '_info.json')
    info_dict = json.load(f)
    f.close()
    
    #iterate through bitmasks, calculate annotation and add to dictionary
    print('Creating annotations...')
    for i,camera in enumerate(coco_dict):
        #indexes for the training and val cutoffs
        train_max_index = info_dict[camera] * (train_ratio)
        print("Adding " + str(train_max_index) + " images to train, rest to val...")
        len_coco = len(coco_dict)
        for j,img_id in enumerate(coco_dict[camera]):
            len_cam = len(coco_dict[camera])
            print("Camera: " + str(i+1) + "/" + str(len_coco) + " | Image: " + str(j+1) + "/" + str(len_cam), flush=True)
            img_dict = coco_dict[camera][img_id]
            if (i > train_max_index):
                val_dict["annotations"].append(img_dict['mask'])
                val_dict["images"].append(img_dict['img'])
            else:
                train_dict["annotations"].append(img_dict['mask'])
                train_dict["images"].append(img_dict['img']) 
    print('Annotaions done!')

    if (not os.path.exists("./Annotations/" + output_dir)):
        os.mkdir("./Annotations/" + output_dir)
    #write dictionaries to files
    print("Writing annotation files...")
    if TEST:
        f = open("./Annotations/" + output_dir + "/test_annotations.json", "w")
    else:
        f = open("./Annotations/" + output_dir + "/train_annotations.json", "w")
        f.write(json.dumps(train_dict))
        f.close()
        f = open("./Annotations/" + output_dir + "/val_annotations.json", "w")
        f.write(json.dumps(val_dict))
        f.close()

    print('OK')
