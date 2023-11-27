import argparse
import os
from skimage import measure
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
import json
import math


def create_mask_annotation(image_path,APPROX):
    image = image_path#ski.io.imread(image_path)
    contour_list = measure.find_contours(image, positive_orientation='low')
    segmentations = []
    polygons = []
    for contour in contour_list:
        for i in range(len(contour)):
            row,col = contour[i]
            contour[i] = (col-1,row-1)
        
        poly = Polygon(contour)
        if(poly.area <= 1): continue

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

    return segmentations, bbox, poly.area


    

if __name__ == "__main__":
    #np.set_printoptions(threshold=np.inf)
    # parse arguments from command line
    parser = argparse.ArgumentParser(description="This program transforms images and their bit masks to COCO dataset formatted segmentation annotations.")
    parser.add_argument("--parent_path", help="This is the path to the parent folder of all the scenes containing training data", required=True, type=str)
    parser.add_argument("--amodal", help="Write true if you want to calculate the amodal masks, default is modal", required=False, type=bool, default=False)
    parser.add_argument("--approx", help="type in True if you wish for the bitmasks to be approximated for a smoother image", required=False, default=False,type=int)
    parser.add_argument("--limit", help="If you wish to not process all images in the path you can select a limit", required=False, default=None,type = int)
    parser.add_argument("--test", help="If test folder should be processed, not train folder", required=False, default=False,type = bool)
    args = parser.parse_args()
    parent_path = args.parent_path
    APPROX = args.approx
    LIMIT = args.limit
    AMODAL = args.amodal

    if args.test:
        mvpsp_succ = 'test'
    else: 
        mvpsp_succ = 'train'

    if not os.path.isdir('Annotations'):
        os.mkdir('Annotations')

    FILTER = True
    # currently the arg parent path should point to mvpsp so that access to /test is also possible
    parent_path = parent_path + '/' + mvpsp_succ 

    #status print
    print("Working on directory: " + parent_path)

    parentDirList = sorted(os.listdir(parent_path))
    len_parentDirList = len(parentDirList)
    id = 1
    coco_dict = {}
    missed_images = []
    missed_bitmasks = []
    for cameraNr,camera in enumerate(parentDirList):

        # If Amodal mask is requested then guide to mask_visib folder, otherwise to mask
        bitmasks_path = parent_path + '/' + camera + ('/mask_visib' if AMODAL else '/mask')
        images_path = parent_path + '/' + camera + '/rgb'
        
        coco_dict[camera] = {}
        
        #create a list of all bitmasks and filter the powerdrill images, 
        #then make sure only those images that have corresponding masks are included in training annotation
        print('Filtering folder: ' + camera + ' | Progress: ' + str(cameraNr + 1) + '/' + str(len_parentDirList))
        #bitmaskDirList = sorted(os.listdir(bitmask_path))
        #imageDirList = sorted(os.listdir(image_path))
        #len_bitmaskDirList = len(bitmaskDirList)
        #len_imageDirList = len(imageDirList)
        
        gt_json_file = open(parent_path + '/' + camera + '/scene_gt.json')
        gt_dict = json.load(gt_json_file)
        bitMaskList = []
        for image in gt_dict:
            for i,bitmask in enumerate(gt_dict[image]):
                # Powerdrill and Screwdriver have ids 1 and 2
                if bitmask['obj_id'] in [1,2]:
                    bitMaskList.append((image,i,bitmask['obj_id'])) # e.g: 001050_000001 becomes (1050,1)
        print('Filtering for ' + camera + ' done!')   

        #iterate through bitmasks, calculate annotation and add to dictionary
        print('Calculating Polygon vertices for COCO Dataset...')
        len_bitMaskList = len(bitMaskList)
        for i,(img, bitmask, object_id) in enumerate(bitMaskList):
            print('Polygon calculation progress: ' + str(i) + '/' + str(len_bitMaskList))
            img_id = str(img)
            bitmask_id = str(bitmask)
            for i in range(0, 6-len(img_id)):
                img_id = '0' + img_id
            for i in range(0, 6-len(bitmask_id)):
                bitmask_id = '0' + bitmask_id

            complete_id = img_id + '_' + bitmask_id
            bitmask_path = bitmasks_path + '/' + complete_id + '.png'
            image_path = parent_path + "/" + camera + "/rgb/" + img_id + ".png"
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
            #from now on we can assume that this image exists
            coco_dict[camera][id] = {}

            # start calculating masks and prepare the json dicts
            temp.convert("1")
            width, height = temp.size
            #add padding to bitmask because find_contours from skimage doesnt account for edge pixels, maybe opencv could be better for this
            bitmask_curr = Image.new("1", (width+2,height+2), 0)
            bitmask_curr.paste(temp, (1,1))
            mask_dict = {}
            mask_dict["segmentation"], mask_dict["bbox"], mask_dict["area"] = create_mask_annotation(np.array(bitmask_curr), APPROX)
            mask_dict["iscrowd"] = 0
            mask_dict["image_id"] = id
            mask_dict["category_id"] = object_id
            mask_dict["id"] = id
            img_dict = {}
            img_dict['id'] = id
            img_dict['width'] = width
            img_dict['height'] = height
            img_dict['file_name'] = (images_path.split('mvpsp/')[1]) + '/' + img_id + '.png'
            coco_dict[camera][id]["img"] = img_dict
            coco_dict[camera][id]["mask"] = mask_dict
            id += 1
    print('Polygons and annotaions done!')

    #write dictionaries to files
    f = open("Annotations/" + mvpsp_succ + "_coco.json", "w")
    f.write(json.dumps(coco_dict))
    f.close()


    f = open("Annotations/" + mvpsp_succ + "_coco_info.json", "w")
    info_dict = {}
    #f.write("camera_id : nr. images in that camera_id\n")
    for camera in coco_dict:
        #f.write(camera + " : " + str(len(coco_dict[camera])) + '\n')
        info_dict[camera] = len(coco_dict[camera])
    f.write(json.dumps(info_dict))
    f.close()

    f = open("Annotations/all_coco_missed.txt", "w")
    f.write(str(missed_images))
    f.write("\n")
    f.write(str(missed_bitmasks))
    f.close()

    print('OK')
