import argparse
import os
from skimage import measure
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
import json
import math

def filterImages(x):
    gt_json = open(globals()['parent_path'] + '/')
    powerdrill_id = "000000" # we are only interested in the powerdrill bitmask for now
    check = x.split("_")[1]
    check = check.split(".")[0]
    print('BitMask ' + str(globals()["cur_bitMask"]) + '/' + str(len_bitmaskDirList))
    globals()["cur_bitMask"] += 1
    if check == powerdrill_id:
        globals()["bitList"].append(x.split("_")[0])
        return True
    else:
        return False
    
def filterImageDirList(x):
    print('Filtering image ' + str(globals()["cur_filterImage"]) + '/' + str(len_imageDirList))
    globals()["cur_filterImage"] += 1   
    if x.split(".png")[0] in globals()["bitList"]:
        return True
    else:
        return False

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
    args = parser.parse_args()
    parent_path = args.parent_path
    APPROX = args.approx
    LIMIT = args.limit
    AMODAL = args.amodal
    
    #status print
    print("Working on directory: " + parent_path)

    if not os.path.isdir('Annotations'):
        os.mkdir('Annotations')

    FILTER = True
    # currently the arg parent path should point to mvpsp so that access to /test is also possible
    parent_path = parent_path + '/train' 
    parentDirList = sorted(os.listdir(parent_path))[:1]
    len_parentDirList = len(parentDirList)
    id = 1
    coco_dict = {}
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
        for i,(img, bitmask, object_id) in enumerate(bitMaskList[:100]):
            print('Polygon calculation progress: ' + str(i) + '/' + str(len_bitMaskList))
            img_id = str(img)
            bitmask_id = str(bitmask)
            for i in range(0, 6-len(img_id)):
                img_id = '0' + img_id
            for i in range(0, 6-len(bitmask_id)):
                bitmask_id = '0' + bitmask_id
            complete_id = img_id + '_' + bitmask_id
            bitmask_path = bitmasks_path + '/' + complete_id + '.png'
            try:
                temp = Image.open(bitmask_path)
            except FileNotFoundError:
                continue
            #from now on we can assume that this image exists
            coco_dict[camera][img_id] = {}

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
            img_dict['file_name'] = (images_path.split('mvpsp/')[1]) + '/' + img_id  
            coco_dict[camera][img_id]["img"] = img_dict
            coco_dict[camera][img_id]["mask"] = mask_dict
            id += 1
    print('Polygons and annotaions done!')

    #write dictionaries to files
    f = open("Annotations/all_coco.json", "w")
    f.write(json.dumps(coco_dict, indent=3))
    f.close()

    print('OK')
