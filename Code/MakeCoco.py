import argparse
import os
from skimage import measure
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
import json
import math



#Values to split the dataset into training, validation and testing
#all scenes are around 12500 images, split into 0.7 train, 0.15 val and 0.15 test
#train and val get dictionaries in coco format, test images are simply listed in the last dict
train_max_index = 8750
val_max_index = 10625

def filterPowerDrill(x):
    powerdrill_id = "000000" # we are only interested in the powerdrill bitmask for now
    check = x.split("_")[1]
    check = check.split(".")[0]
    print('BitMask ' + str(globals()["cur_bitMask"]) + '/' + str(len_bitMaskDirList))
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

# def create_mask_annotation(image_path):
#     image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
#     if image is None:
#         sys.exit("ERROR: Could not read the image.")
#     ret, binary = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
#     contours = []
#     contours = cv.findContours(binary, contours, cv.CHAIN_APPROX_SIMPLE)
#     cv.imshow("display", binary)
#     k = cv.waitKey(0)

def create_mask_annotation(image_path,APPROX):
    image = image_path#ski.io.imread(image_path)
    contour_list = measure.find_contours(image, positive_orientation='low')
    segmentations = []
    polygons = []
    for contour in contour_list:
        for i in range(len(contour)):
            row,col = contour[i]
            contour[i] = (col-1,row-1)
        
        #fig, ax = plt.subplots()
        #ax.plot(contour[:,0], contour[:,1])
        #plt.show()
        
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
    parser.add_argument("--image_parent", help="This is the path to the parent folder of all the scenes containing training data", required=True, type=str)
    parser.add_argument("--amodal", help="Write true if you want to calculate the amodal masks, default is modal", required=False, type=bool, default=False)
    parser.add_argument("--approx", help="type in True if you wish for the bitmasks to be approximated for a smoother image", required=False, default=False,type=int)
    parser.add_argument("--limit", help="If you wish to not process all images in the path you can select a limit", required=False, default=None,type = int)
    args = parser.parse_args()
    parent_path = args.image_parent
    APPROX = args.approx
    LIMIT = args.limit
    AMODAL = args.amodal
    
    #status print
    print("Working on directory: " + parent_path)

    if not os.path.isdir('Annotations'):
        os.mkdir('Annotations')
    
    #needed global variable for status updates
    cur_bitMask = 1
    cur_filterImage = 1
    bitList = []

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

    # Sometimes the drill is not in the scene, 
    # this can be incorporated in a useful manner at some later point in time
    FILTER = True
    cameras = {}
    parentDirList = sorted(os.listdir(parent_path))
    len_parentDirList = len(parentDirList)
    id = 1
    for cameraNr,camera in enumerate(parentDirList):
        cameras[camera] = {}
        cameras[camera]['bitmasks'] = []

        # If Amodal mask is requested then guide to mask_visib folder, otherwise to mask
        bitmask_path = parent_path + '/' + camera + ('/mask_visib' if AMODAL else '/mask')
        image_path = parent_path + '/' + camera + '/rgb'

        #create a list of all bitmasks and filter the powerdrill images, 
        #then make sure only those images that have corresponding masks are included in training annotation
        print('Filtering the powerdrill in the bitmasks...')
        print('Filtering camera ' + str(cameraNr) + '/' + str(len_parentDirList))
        bitMaskDirList = os.listdir(bitmask_path)
        imageDirList = os.listdir(image_path)
        len_bitMaskDirList = len(bitMaskDirList)
        len_imageDirList = len(imageDirList)
        bitmaskDirList = list(filter(filterPowerDrill, bitMaskDirList))
        bitmaskDirList = sorted(bitmaskDirList)
        if FILTER:
            print('Filtering the images to only the ones with corresponding bitmasks...')
            imageDirList = list(filter(filterImageDirList, imageDirList))
            imageDirList = sorted(imageDirList)
        else:
            imageDirList = sorted(os.listdir(image_path))
        print('Filtering Done!')
             

        #check that both directories have same length
        #len_bitMaskDirList = len(bitMaskDirList)
        #len_imageDirList = len(imageDirList)
        #assert(len_imageDirList == len_bitMaskDirList)

        #reset global variables necessary for status printing in helper functions
        cur_bitMask = 1
        bitList = []   
        cur_filterImage = 1

        

        #iterate through bitmasks, calculate annotation and add to dictionary
        print('Calculating Polygon vertices for COCO Dataset...')
        for i,img_path in enumerate(bitmaskDirList[:LIMIT]):
            current_path = bitmask_path + "/" + img_path
            temp = Image.open(current_path)
            temp.convert("1")
            width, height = temp.size
            #add padding to bitmask because find_contours from skimage doesnt account for edge pixels, maybe opencv could be better for this
            bitmask_curr = Image.new("1", (width+2,height+2), 0)
            bitmask_curr.paste(temp, (1,1))
            mask_dict = {}
            mask_dict["segmentation"], mask_dict["bbox"], mask_dict["area"] = create_mask_annotation(np.array(bitmask_curr), APPROX)
            mask_dict["iscrowd"] = 0
            mask_dict["image_id"] = id
            mask_dict["category_id"] = 1
            mask_dict["id"] = id
            img_dict = {}
            img_dict['id'] = id
            img_dict['width'] = width
            img_dict['height'] = height
            img_dict['file_name'] = camera + '_' + imageDirList[i]
            if (i > val_max_index):
                test_dict["annotations"].append(mask_dict)
                test_dict["images"].append(img_dict)
            elif (i > train_max_index):
                val_dict["annotations"].append(mask_dict)
                val_dict["images"].append(img_dict)
            else:
                train_dict["annotations"].append(mask_dict)
                train_dict["images"].append(img_dict) 
            id += 1
    print('Polygons and annotaions done!')
    

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
