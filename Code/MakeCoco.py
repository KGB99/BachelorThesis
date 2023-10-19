import argparse
import os
from skimage import measure
import skimage as ski
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon

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
    for contour in contour_list:
        for i in range(len(contour)):
            row,col = contour[i]
            contour[i] = (col-1,row-1)
        
        poly = Polygon(contour)
        if(poly.area <= 1): continue

        if APPROX:
            poly = poly.simplify(1.0, preserve_topology=False)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        x, y, max_x, max_y = poly.bounds
        bbox = (x,y,max_x-x,max_y-y)
    return segmentation, bbox, poly.area


    

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    # parse arguments from command line
    parser = argparse.ArgumentParser(description="This program transforms images and their bit masks to COCO dataset formatted segmentation annotations.")
    parser.add_argument("--path_image", help="This is the path to the folder containing the images", required=True, type=str)
    parser.add_argument("--path_bitmask", help="This is the path to the folder containing the bitmasks of the images", required=True, type=str)
    parser.add_argument("--approx", help="type in True if you wish for the bitmasks to be approximated for a smoother image", required=False, default=False,type=int)
    parser.add_argument("--limit", help="If you wish to not process all images in the path you can select a limit", required=False, default=0,type = int)
    args = parser.parse_args()
    image_path = args.path_image 
    APPROX = args.approx
    bitmask_path = args.path_bitmask
    LIMIT = args.limit
    FILTER = True
    
    bitmaskDirList = list(filter(filterPowerDrill, sorted(os.listdir(bitmask_path))))
    if FILTER:
        imageDirList = list(filter(filterImageDirList, sorted(os.listdir(image_path))))
    else:
        imageDirList = sorted(os.listdir(image_path))
    annotations = []
    for i,img_path in enumerate(bitmaskDirList):
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
        current_dict["image_id"] = imageDirList[i].split(".png")[0]
        current_dict["category_id"] = 1
        current_dict["id"] = i
        annotations.append(current_dict)
    
    if (len(annotations) != len(bitmaskDirList) != len(imageDirList)):
        print("Lengths of directories and bitmasks dont match!")