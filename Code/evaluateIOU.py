import cv2
import os 
import json 
import numpy as np
import torch
import ast
import pycocotools


if __name__ == '__main__':
    bbox_preds_path = '/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/results/bbox_detections.json'
    mask_preds_path = '/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/results/mask_detections.json'
    test_annotations_path = '/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/test_annotations.json'
    output = '/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/predicted_video.mp4'

    # read test ground truth annotations
    f = open(test_annotations_path, 'r')
    test_annotations_dict = json.load(f)
    f.close()
    
    bbox_dict = {}
    f = open(bbox_preds_path, 'r')
    line_list = ast.literal_eval(f.readline())
    for line_dict in line_list:
        curr_id = line_dict['image_id']
        curr_cat = line_dict['category_id']
        bbox_dict[curr_id] = {}
        bbox_dict[curr_id][curr_cat] = {}
        bbox_dict[curr_id][curr_cat]['bbox'] = line_dict['bbox']
        bbox_dict[curr_id][curr_cat]['score'] = line_dict['score']
    f.close()

    
    mask_dict = {}
    f = open(mask_preds_path, 'r')
    line_list = ast.literal_eval(f.readline())
    for line_dict in line_list:
        curr_id = line_dict['image_id']
        curr_cat = line_dict['category_id']
        
        print(line_dict)
        exit()
        mask_dict[curr_id] = {}
        mask_dict[curr_id][curr_cat] = {}
        mask_dict[curr_id][curr_cat]['bbox' ] = line_dict['bbox']
        mask_dict[curr_id][curr_cat]['score'] = line_dict['score']
    f.close()
    
    # create mapping for img_id -> img_path
    img_mappings = {}
    path_prepend = '/Users/kerim/Desktop/'
    for img_dict in test_annotations_dict['images']:
        img_mappings[img_dict['id']] = path_prepend + img_dict['file_name']

    #print(len(test_annotations_dict['annotations']))
    bboxes_found = 0
    bbox_avg_accuracy = 0
    iou_total = 0
    total_bboxes = len(test_annotations_dict['annotations'])

    # prepare video sequence
    height, width, channels = (1080, 1280, 3)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for gt_dict in test_annotations_dict['annotations']:
        # load image
        img_path = img_mappings[gt_dict['image_id']]
    
        gt_boxes = gt_dict['bbox']
        gt_segmentation = gt_dict['segmentation']
        gt_img_id = gt_dict['image_id']
        gt_cat_id = gt_dict['category_id']

        # load image
        img_path = img_mappings[gt_dict['image_id']]
        image = cv2.imread(img_path)
        

        # load predicted labels
        try:
            pred_bbox = bbox_dict[gt_img_id][gt_cat_id]['bbox']
            
        except KeyError:
            print('Bounding Box not found!')
            continue
        
        # point1(x,y) = (pr_x,pr_y) is format of writing in bbox detections
        pr_x = int(pred_bbox[0])
        pr_y = int(pred_bbox[1])
        pr_w = int(pred_bbox[2])
        pr_h = int(pred_bbox[3])
        pr_area = pr_w * pr_h
 
        gt_x = int(gt_boxes[0])
        gt_y = int(gt_boxes[1])
        gt_w = int(gt_boxes[2])
        gt_h = int(gt_boxes[3])
        gt_area = gt_w * gt_h

        i_x0 = max(pr_x, gt_x)
        i_y0 = max(pr_y, gt_y)
        i_x1 = min(pr_x+pr_h, gt_x + gt_h)
        i_y1 = min(pr_y + pr_w, gt_x + gt_w)
        intersection_area = (i_x1 - i_x0) * (i_y1 - i_y0)

        # paint overlays of both rectangles using cv2
        """
        if (img_path == '/Users/kerim/Desktop/test/004000/rgb/000286.png'):
            print('pr_p0: ' + str((pr_x,pr_y)))
            print('pr_(w,h): ' + str((pr_w,pr_h)))
            print('gt_p0: ' + str((gt_x,gt_y)))
            print('gt_(w,h): ' + str((gt_w,gt_h)))
            print('i_p0: ' + str((i_x0,i_y0)))  
            print('i_p1: ' + str((i_x1, i_y1)))
        """

        cv2.rectangle(image, (pr_x, pr_y), (pr_x + pr_w, pr_y + pr_h), (0,0,255),1)
        cv2.rectangle(image, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0,255,0),1)
        #cv2.imshow(img_path,image)
        #cv2.waitKey(0)
        out.write(image)

        union_area = gt_area + pr_area - intersection_area

        iou = intersection_area/union_area
        print(iou)
        iou_total += iou

        bboxes_found = bboxes_found + 1
    out.release()
    cv2.destroyAllWindows()
    print('Percentage of bounding boxes detected: ' + str(bboxes_found/total_bboxes))
    print('Average Intersection over union: ' + str(iou_total / bboxes_found))
        