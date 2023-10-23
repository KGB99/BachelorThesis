import argparse
import os
from skimage import measure
import skimage as ski
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
import shapely
import json

bitList = []

def filterPowerDrill(x):
    powerdrill_id = "000000" # we are only interested in the powerdrill bitmask for now
    check = x.split("_")[1]
    check = check.split(".")[0]
    if check == powerdrill_id:
        bitList.append(x.split("_")[0])
        return True
    else:
        return False
    
def filterImageDirList(x):
    if x.split(".png")[0] in bitList:
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

    # Sometimes the drill is not in the scene, 
    # this can be incorporated in a useful manner at some later point in time
    FILTER = True
    cameras = {}
    print(parent_path)
    parentDirList = sorted(os.listdir(parent_path))
    for camera in parentDirList:
        cameras[camera] = {}
        cameras[camera]['bitmasks'] = []
        # If Amodal mask is requested then guide to mask_visib folder, otherwise to mask
        bitmask_path = camera + ('/mask_visib' if AMODAL else '/mask')
        print(camera)
        
    exit()

    
    bitmaskDirList = list(filter(filterPowerDrill, sorted(os.listdir(bitmask_path))))
    if FILTER:
        imageDirList = list(filter(filterImageDirList, sorted(os.listdir(image_path))))
    else:
        imageDirList = sorted(os.listdir(image_path))
    coco_dataset = {} 
    coco_dataset["annotations"] = []

    for i,img_path in enumerate(bitmaskDirList[:LIMIT]):
        current_path = bitmask_path + "/" + img_path
        temp_curr = Image.open(current_path)
        temp_curr.convert("1")
        width, height = temp_curr.size
        #add padding to bitmask because find_contours from skimage doesnt account for edge pixels, maybe opencv could be better for this
        bitmask_curr = Image.new("1", (width+2,height+2), 0)
        bitmask_curr.paste(temp_curr, (1,1))
        current_dict = {}
        current_dict["segmentation"], current_dict["bbox"], current_dict["area"] = create_mask_annotation(np.array(bitmask_curr), APPROX)
        current_dict["iscrowd"] = 0
        current_dict["image_id"] = int(imageDirList[i].split(".")[0])
        current_dict["category_id"] = 1
        current_dict["id"] = int(imageDirList[i].split(".")[0])
        coco_dataset["annotations"].append(current_dict)

    coco_dataset["info"] = {"description" : "COCO dataset annotations for the medical dataset from cvg's Jonas Hein"}
    coco_dataset["licenses"] = {}
    coco_dataset["images"] = []
    for i,image in enumerate(imageDirList[:LIMIT]):
        curr_annotation = {}
        curr_annotation["id"] = int(image.split(".")[0])
        curr_annotation["width"] = width
        curr_annotation["height"] = height
        curr_annotation["file_name"] = image
        coco_dataset["images"].append(curr_annotation)
    
    annotation_file = open("dataset_coco.json", "w")
    annotation_file.write(json.dumps(coco_dataset, indent=3))
    annotation_file.close()
