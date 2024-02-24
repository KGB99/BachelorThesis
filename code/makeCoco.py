import argparse
import os
from skimage import measure
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
import json
import time


def create_mask_annotation(image_path,APPROX):
    image = image_path#ski.io.imread(image_path)
    contour_list = measure.find_contours(image, positive_orientation='low')
    segmentations = []
    polygons = []
    poly = -1
    bbox = -1
    for contour in contour_list:
        for i in range(len(contour)):
            row,col = contour[i]
            contour[i] = (col-1,row-1)
        
        poly = Polygon(contour)
        if (poly.area <= 1): continue

        if APPROX:
            poly = poly.simplify(1.0, preserve_topology=False)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        #coords = np.array(poly.exterior.coords)
        #fig, ax = plt.subplots()
        #ax.plot(coords[:,0],coords[:,1])
        #plt.show()
        segmentations.append(segmentation)
        polygons.append(poly)
        
        multipoly = MultiPolygon(polygons)
        x1, y1, x2, y2 = multipoly.bounds
        bbox = (x1, y1, x2-x1, y2-y1)
    if (bbox == -1) or (poly == -1):
        return -1,-1,-1
    return segmentations, bbox, poly.area

def createCocoFromMultipleFolders(args):
    parent_folders = eval(args.folders)
    parent_path = args.parent_path
    output_file = args.output_file
    output_dir = args.output_dir
    write_missed = args.write_missed
    image_file_ending = args.img_file_type
    bitmask_file_ending = args.bitmask_file_type
    APPROX = args.approx
    LIMIT_IMAGES = args.limit_images
    AMODAL = args.amodal
    LIMIT_FOLDERS = args.limit_folder
    path_splitter = args.path_splitter + '/'
    stride = args.stride

    #status print
    print("Path: " + parent_path)
    print("Directories: " + str(parent_folders))   
    len_parent_folders = len(parent_folders)

    # if parent folders are empty then just go through all the folders in parent_path
    if len_parent_folders == 0:
        print("No folders given in parent_folders argument, so going through all of them...")
        parent_folders = os.listdir(parent_path)

    id = 1
    coco_dict = {}
    if write_missed:
        missed_images = []
        missed_bitmasks = []
    for folderNr, folder in enumerate(parent_folders):
        subFolders = os.listdir(parent_path + "/" + folder)
        len_subFolders = len(subFolders)
        for cameraNr, camera in enumerate(subFolders):
            curr_folder_path = parent_path + "/" + folder
            print("Currently Processing: " + folder + "/" + camera)
                
            # If Amodal mask is requested then guide to mask_visib folder, otherwise to mask
            bitmasks_path = curr_folder_path + '/' + camera + ('/mask_visib' if AMODAL else '/mask')
            images_path = curr_folder_path + '/' + camera + '/rgb'

            coco_dict[folder + "/" + camera] = {}
            
            #create a list of all bitmasks and filter the powerdrill images, 
            #then make sure only those images that have corresponding masks are included in training annotation
            #bitmaskDirList = sorted(os.listdir(bitmask_path))
            #imageDirList = sorted(os.listdir(image_path))
            #len_bitmaskDirList = len(bitmaskDirList)
            #len_imageDirList = len(imageDirList)
            
            gt_json_file = open(curr_folder_path + '/' + camera + '/scene_gt.json')
            gt_dict = json.load(gt_json_file)
            bitMaskList = []
            for image in gt_dict:
                for i,bitmask in enumerate(gt_dict[image]):
                    # Powerdrill and Screwdriver have ids 1 and 2
                    if bitmask['obj_id'] in [1,2]:
                        bitMaskList.append((image,i,bitmask['obj_id'])) # e.g: 001050_000001 becomes (1050,1)
            print('Filtering for ' + folder + "/" + camera + ' done!')   

            #iterate through bitmasks, calculate annotation and add to dictionary
            print('Calculating Polygon vertices for COCO Dataset...')
            len_bitMaskList = len(bitMaskList)
            for i,(img, bitmask, object_id) in enumerate(bitMaskList):
                print('Progress: Folder=' + str(folderNr+1) + '/' + str(len_parent_folders) + ' | Camera=' + str(cameraNr+1) + '/' + str(len_subFolders) + ' | Image=' + str(i+1) + '/' + str(len_bitMaskList), flush=True)
                img_id = str(img)
                bitmask_id = str(bitmask)
                for i in range(0, 6-len(img_id)):
                    img_id = '0' + img_id
                for i in range(0, 6-len(bitmask_id)):
                    bitmask_id = '0' + bitmask_id

                complete_id = img_id + '_' + bitmask_id
                bitmask_path = bitmasks_path + '/' + complete_id + '.' + bitmask_file_ending
                image_path = images_path + "/" + img_id + "." + image_file_ending
                print('Calculating: ' + image_path)
                if (not os.path.exists(image_path)):
                    print("Image " + img_id + "." + image_file_ending + " does not exist at " + image_path)
                    if write_missed:
                        missed_images.append(image_path)
                    continue
                if (not os.path.exists(bitmask_path)):
                    print("Bitmask at " + bitmask_path + " does not exist")
                    if write_missed:
                        missed_bitmasks.append(bitmask_path)
                    continue
                
                try:
                    temp = Image.open(bitmask_path)
                except:
                    print("Error opening bitmask at: " + bitmask_path)
                    continue

                # start calculating masks and prepare the json dicts
                temp.convert("1")
                width, height = temp.size
                #add padding to bitmask because find_contours from skimage doesnt account for edge pixels, maybe opencv could be better for this
                bitmask_curr = Image.new("1", (width+2,height+2), 0)
                bitmask_curr.paste(temp, (1,1))
                mask_dict = {}
                try:
                    mask_dict["segmentation"], mask_dict["bbox"], mask_dict["area"] = create_mask_annotation(np.array(bitmask_curr), APPROX)
                    if mask_dict["segmentation"] == -1 or mask_dict["bbox"] == -1 or mask_dict["area"] == -1:
                        print("No bitmask properly found at image: " + image_path)
                        continue
                except Exception as e:
                    print("Exception: " + str(e) + " at image: " + image_path)
                    continue
                mask_dict["iscrowd"] = 0
                mask_dict["image_id"] = id
                mask_dict["category_id"] = object_id
                mask_dict["id"] = id
                img_dict = {}
                img_dict['id'] = id
                img_dict['width'] = width
                img_dict['height'] = height
                img_dict['file_name'] = (images_path.split(path_splitter)[1]) + '/' + img_id + '.' + image_file_ending

                #from now on we can assume that this image exists
                coco_dict[folder + "/" + camera][id] = {}
                coco_dict[folder + "/" + camera][id]["img"] = img_dict
                coco_dict[folder + "/" + camera][id]["mask"] = mask_dict
                id += 1
    print('Polygons and annotaions done!')
    print("Writing output files...")

    #write dictionaries to files

    if not os.path.isdir('Annotations/' + output_dir):
        os.mkdir('Annotations/' + output_dir)
    
    f = open("Annotations/" + output_dir + "/" + output_file + ".json", "w")
    f.write(json.dumps(coco_dict))
    f.close()


    f = open("Annotations/" + output_dir + "/" + output_file + "_info.json", "w")
    info_dict = {}
    #f.write("camera_id : nr. images in that camera_id\n")
    for camera in coco_dict:
        #f.write(camera + " : " + str(len(coco_dict[camera])) + '\n')
        info_dict[camera] = len(coco_dict[camera])
    f.write(json.dumps(info_dict))
    f.write("\n")
    f.close()

    if write_missed:
        f = open("Annotations/" + output_dir + "/" + output_file + "_missed.txt", "w")
        f.write(str(missed_images))
        f.write("\n")
        f.write(str(missed_bitmasks))
        f.close()

def createCocoFromSingleFolder(args):
    parent_folders = eval(args.folders)
    parent_path = args.parent_path
    output_file = args.output_file
    output_dir = args.output_dir
    write_missed = args.write_missed
    image_file_ending = args.img_file_type
    bitmask_file_ending = args.bitmask_file_type
    APPROX = args.approx
    LIMIT_IMAGES = args.limit_images
    AMODAL = args.amodal
    LIMIT_FOLDERS = args.limit_folder
    path_splitter = args.path_splitter + '/'
    stride = args.stride


    #status print
    print("Path: " + parent_path)
    print("Directories: " + str(parent_folders))   
    len_parent_folders = len(parent_folders)
    
    # if parent folders are empty then just go through all the folders in parent_path
    if len_parent_folders == 0:
        print("No folders given in parent_folders argument, so going through all of them...")
        parent_folders = os.listdir(parent_path)

    id = 0
    coco_dict = {}
    missed_images = []
    missed_bitmasks = []

    for cameraNr, camera in enumerate(parent_folders):
        curr_folder_path = parent_path 
        print(curr_folder_path)
        print("Currently Processing Folder: " + camera + " || Folder Progress: " + str(cameraNr + 1) + "/" + str(len_parent_folders))
            
        # If Amodal mask is requested then guide to mask_visib folder, otherwise to mask
        bitmasks_path = curr_folder_path + '/' + camera + ('/mask_visib' if AMODAL else '/mask')
        images_path = curr_folder_path + '/' + camera + '/rgb'
        
        coco_dict[camera] = {}
        
        #create a list of all bitmasks and filter the powerdrill images, 
        #then make sure only those images that have corresponding masks are included in training annotation
        #bitmaskDirList = sorted(os.listdir(bitmask_path))
        #imageDirList = sorted(os.listdir(image_path))
        #len_bitmaskDirList = len(bitmaskDirList)
        #len_imageDirList = len(imageDirList)
        
        gt_json_file = open(curr_folder_path + '/' + camera + '/scene_gt.json')
        gt_dict = json.load(gt_json_file)
        bitMaskList = []
        for image in gt_dict:
            for i,bitmask in enumerate(gt_dict[image]):
                # Powerdrill id = 1and Screwdriver id = 2
                #if bitmask['obj_id'] in [1,2]: this has been moved down to the main loop so that i can add the gt_exists option to dicts for false positives in evaluation
                bitMaskList.append((image,i,bitmask['obj_id'])) # e.g: 001050_000001 becomes (1050,1)
        print('Filtering for ' + camera + ' done!')   

        #iterate through bitmasks, calculate annotation and add to dictionary
        print('Calculating Polygon vertices for COCO Dataset...')
        len_bitMaskList = len(bitMaskList)
        for i,(img, bitmask, object_id) in enumerate(bitMaskList):
            id += 1
            if object_id not in [1,2]:
                img_dict = {}
                img_dict['id'] = id
                img_dict['width'] = width
                img_dict['height'] = height
                img_dict['file_name'] = (images_path.split(path_splitter)[1]) + '/' + img_id + '.' + image_file_ending
                coco_dict[camera][id] = {}
                coco_dict[camera][id]["gt_exists"] = 0
                coco_dict[camera][id]["img"] = img_dict
                coco_dict[camera][id]["mask"] = []
                continue

            if ((i % stride) != 0):
                continue
            print('Progress: Camera=' + str(cameraNr + 1) + '/' + str(len_parent_folders) + ' | Image=' + str(i + 1) + '/' + str(len_bitMaskList) + ' | Bitmask=' + str((i%2) + 1) + '/2')
            img_id = str(img)
            bitmask_id = str(bitmask)
            for i in range(0, 6-len(img_id)):
                img_id = '0' + img_id
            for i in range(0, 6-len(bitmask_id)):
                bitmask_id = '0' + bitmask_id

            complete_id = img_id + '_' + bitmask_id
            bitmask_path = bitmasks_path + '/' + complete_id + '.' + bitmask_file_ending
            image_path = curr_folder_path + "/" + camera + "/rgb/" + img_id + "." + image_file_ending
            print('Calculating: ' + image_path, flush=True)
            if (not os.path.exists(image_path)):
                print("Image " + img_id + ".png does not exist at " + image_path)
                missed_images.append(image_path)
                continue
            if (not os.path.exists(bitmask_path)):
                print("Bitmask at " + bitmask_path + " does not exist")
                missed_bitmasks.append(bitmask_path)
                continue
            
            try:
                temp = Image.open(bitmask_path)
            except:
                continue

            # start calculating masks and prepare the json dicts
            temp.convert("1")
            width, height = temp.size
            #add padding to bitmask because find_contours from skimage doesnt account for edge pixels, maybe opencv could be better for this
            bitmask_curr = Image.new("1", (width+2,height+2), 0)
            bitmask_curr.paste(temp, (1,1))
            mask_dict = {}
            try:
                mask_dict["segmentation"], mask_dict["bbox"], mask_dict["area"] = create_mask_annotation(np.array(bitmask_curr), APPROX)
                if (mask_dict["segmentation"] == -1 or mask_dict["bbox"] == -1 or mask_dict["area"] == -1):
                    coco_dict[camera][id] = {}
                    coco_dict[camera][id]["gt_exists"] = 0
                    continue
            except Exception:
                print("EXCEPTION AT IMAGE: " + image_path)
                continue
            mask_dict["iscrowd"] = 0
            mask_dict["image_id"] = id
            mask_dict["category_id"] = object_id
            mask_dict["id"] = id
            img_dict = {}
            img_dict['id'] = id
            img_dict['width'] = width
            img_dict['height'] = height
            img_dict['file_name'] = (images_path.split(path_splitter)[1]) + '/' + img_id + '.' + image_file_ending
            
            #from now on we can assume that this image exists
            coco_dict[camera][id] = {}
            coco_dict[camera][id]["gt_exists"] = 1
            coco_dict[camera][id]["img"] = img_dict
            coco_dict[camera][id]["mask"] = mask_dict
            #id += 1
    print('Polygons and annotaions done!')
    print("Writing output files...")

    #write dictionaries to files

    if not os.path.isdir('Annotations/' + output_dir):
        os.mkdir('Annotations/' + output_dir)
    
    f = open("Annotations/" + output_dir + "/" + output_file + ".json", "w")
    f.write(json.dumps(coco_dict))
    f.close()


    f = open("Annotations/" + output_dir + "/" + output_file + "_info.json", "w")
    info_dict = {}
    #f.write("camera_id : nr. images in that camera_id\n")
    for camera in coco_dict:
        #f.write(camera + " : " + str(len(coco_dict[camera])) + '\n')
        info_dict[camera] = len(coco_dict[camera])
    f.write(json.dumps(info_dict))
    f.write("\n")
    f.close()

    if write_missed:
        f = open("Annotations/" + output_dir + "/" + output_file + "_missed.txt", "w")
        f.write(str(missed_images))
        f.write("\n")
        f.write(str(missed_bitmasks))
        f.close()

    

if __name__ == "__main__":
    #np.set_printoptions(threshold=np.inf)
    # parse arguments from command line
    parser = argparse.ArgumentParser(description="This program transforms images and their bit masks to COCO dataset formatted segmentation annotations.")
    parser.add_argument("--parent_path", help="General parent path in which all chosen folders are in", required=True, type=str)
    parser.add_argument("--multiple_folders", default=False, required=False, type=bool)
    parser.add_argument("--folders", help="List of folders of chosen scenes containing data", required=True, type=str)
    parser.add_argument("--amodal", help="Write true if you want to calculate the amodal masks, default is modal", required=False, type=bool, default=False)
    parser.add_argument("--approx", help="type in True if you wish for the bitmasks to be approximated for a smoother image", required=False, default=False,type=int)
    parser.add_argument("--limit_images", help="If you wish to not process all images in the path you can select a limit", required=False, default=None,type = int)
    parser.add_argument("--limit_folder", help="Limit nr of top-level folders to be processed", required=False, default=0, type=int)
    parser.add_argument("--output_file", help="Name of output file", required=False, default="output", type=str)
    parser.add_argument("--output_dir", help="Name of output dir", required=False, default="output", type=str)
    parser.add_argument("--img_file_type", help="File type of the images e.g: jpg or png", required=True, type=str)
    parser.add_argument("--bitmask_file_type", help="File type of bitmasks e.g: jpg or png", required=True, type=str)
    parser.add_argument("--write_missed", help="option to write the missed images to a file", required=False, default=False, type=bool)
    parser.add_argument("--path_splitter", help="point at which image path gets split and put into the result_dict under 'file_name'", required=False, default='mvpsp', type=str)
    parser.add_argument("--stride", required=False, default=1, type=int)
    args = parser.parse_args()
    

    start = time.time()

    if not os.path.isdir('Annotations'):
        os.mkdir('Annotations')

    if args.multiple_folders:
        createCocoFromMultipleFolders(args)
    else:
        createCocoFromSingleFolder(args)    
    end = time.time()
    print('Time of execution: ' + str(end-start) + 's')
    print('Done working on path:' + args.parent_path + ' on folders: ' + str(args.folders))
    print('OK!')
