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
    parser.add_argument("--test", help="type true if creating testing annotations so it doesnt create two files", required=False, default=False, type=bool)
    parser.add_argument("--val_folders", help="If instead of a ratio you wish to allocate certain folders to validation, then write them here", required=False, type=str, default=[])
    parser.add_argument("--with_val", required=False, type=bool, default=True)
    
    args = parser.parse_args()
    output_dir = args.output_dir
    coco_dir = args.coco_file
    train_ratio = args.train_ratio
    val_folders = args.val_folders
    use_val_folders = (len(val_folders) != 0)
    TEST = args.test
    with_val = args.with_val
    

    if not os.path.isdir('./Annotations'):
        os.mkdir('./Annotations')

    #create dictionary for the annotations in COCO style
    if not TEST:
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
    else:
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
    print("Loading annotation_info file...")
    f = open(coco_dir[::-1].split('.')[1][::-1] + '_info.json')
    info_dict = json.load(f)
    f.close()
    
    #iterate through bitmasks, calculate annotation and add to dictionary
    if not use_val_folders:
        train_to_val_ratio_dict = {}
    else:
        val_folders_confirmation = {}
    print('Creating annotations...')
    for i,camera in enumerate(coco_dict):
        if not use_val_folders:
            #indexes for the training and val cutoffs
            train_max_index = info_dict[camera] * (train_ratio)
            train_to_val_ratio_dict[camera] = (train_max_index, info_dict[camera])
            print("Max index for training " + str(train_max_index))

        len_coco = len(coco_dict)
        for j,img_id in enumerate(coco_dict[camera]):
            #if coco_dict[camera][img_id]["gt_exists"] == 0: 
                # currently doing continue cause this is how i used to do it
                # should try to train with some examples of no ground truths too
                #continue 
            len_cam = len(coco_dict[camera])
            print("Camera: " + str(i+1) + "/" + str(len_coco) + " | Image: " + str(j+1) + "/" + str(len_cam), flush=True)
            img_dict = coco_dict[camera][img_id]
            if not TEST:
                if not use_val_folders:
                    if (with_val):
                        if (j > train_max_index):
                            val_dict["annotations"].append(img_dict['mask'])
                            val_dict["images"].append(img_dict['img'])
                        else:
                            train_dict["annotations"].append(img_dict['mask'])
                            train_dict["images"].append(img_dict['img']) 
                    else:
                        train_dict["annotations"].append(img_dict['mask'])
                        train_dict["images"].append(img_dict['img']) 
                else:
                    if camera in val_folders:
                        val_dict["annotations"].append(img_dict['mask'])
                        val_dict["images"].append(img_dict['img'])
                        if camera not in val_folders_confirmation.keys():
                            val_folders_confirmation[camera] = 0
                        val_folders_confirmation[camera] += 1
                    else:
                        train_dict["annotations"].append(img_dict['mask'])
                        train_dict["images"].append(img_dict['img'])
            else:
                test_dict["annotations"].append(img_dict['mask'])
                test_dict["images"].append(img_dict['img'])
    print('Annotaions done!')

    if (not os.path.exists("./Annotations/" + output_dir)):
        os.mkdir("./Annotations/" + output_dir)
    #write dictionaries to files
    print("Writing annotation files...")
    if TEST:
        f = open("./Annotations/" + output_dir + "/test_annotations.json", "w")
        f.write(json.dumps(test_dict))
        f.close()
    else:
        f = open("./Annotations/" + output_dir + "/train_annotations.json", "w")
        f.write(json.dumps(train_dict))
        f.close()
        if with_val:
            f = open("./Annotations/" + output_dir + "/val_annotations.json", "w")
            f.write(json.dumps(val_dict))
            f.close()
    if not TEST:
        if not use_val_folders:
            print("Ratios of training to validation:")
            print(train_to_val_ratio_dict,flush=True)
        else:
            print("Folders of validation and nr of images in them:")
            print(val_folders_confirmation, flush=True)
    print('OK')
