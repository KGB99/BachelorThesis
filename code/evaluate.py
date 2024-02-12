import cv2
import os 
import json 
import numpy as np
import ast
import pycocotools.mask as maskUtils
from matplotlib import pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--preds_path", required=True, type=str)
    parser.add_argument("--labels_path", required=True, type=str)
    parser.add_argument("--images_dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--video", required=False, type=bool, default=False)
    args = parser.parse_args()
    images_dir_path = args.images_dir
    #/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/results
    bbox_preds_path = args.preds_path + '/bbox_detections.json'
    mask_preds_path = args.preds_path + '/mask_detections.json'
    VIDEO = args.video
    #'/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/test_annotations.json'
    test_annotations_path = args.labels_path
    #'/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/predicted_video.mp4'
    output = './Annotations/' + args.output
    if not os.path.isdir(output):
        os.mkdir(output)
    if VIDEO:
        video_path = output + '/predicted_video.mp4'
    else:
        if not os.path.isdir(output + '/evaluatedImages'):
            os.mkdir(output + '/evaluatedImages')
        output_images_path = output + '/evaluatedImages'

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
        mask_dict[curr_id] = {}
        mask_dict[curr_id][curr_cat] = {}
        mask_dict[curr_id][curr_cat]['segmentation' ] = line_dict['segmentation']
        mask_dict[curr_id][curr_cat]['score'] = line_dict['score']
    f.close()
    
    # create mapping for img_id -> img_path
    img_mappings = {}
    path_prepend = images_dir_path + '/'
    for img_dict in test_annotations_dict['images']:
        img_mappings[img_dict['id']] = path_prepend + img_dict['file_name']

    #print(len(test_annotations_dict['annotations']))
    bboxes_found = 0
    bbox_avg_accuracy = 0
    iou_bbox_total = 0
    seg_found = 0
    iou_seg_total = 0


    # prepare video sequence
    height, width, channels = (1080, 1280, 3)
    if VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

    total_dets = 0
    for i,gt_dict in enumerate(test_annotations_dict['annotations']):
        total_dets += 1
        # load image
        print("Processing Image: " + img_mappings[gt_dict['image_id']])
    
        gt_boxes = gt_dict['bbox']
        gt_seg_vertices = gt_dict['segmentation']
        gt_img_id = gt_dict['image_id']
        gt_cat_id = gt_dict['category_id']

        # load image
        img_path = img_mappings[gt_dict['image_id']]
        image = cv2.imread(img_path)
        h,w,_ = image.shape
        

        # load predicted labels
        try:
            pred_bbox = bbox_dict[gt_img_id][gt_cat_id]['bbox']
            pred_seg = mask_dict[gt_img_id][gt_cat_id]['segmentation']
        except KeyError:
            print('Bounding Box or Segmentation not found!')
            continue
        
        # decode masks
        pred_mask = maskUtils.decode(pred_seg)
        #pred_mask = pred_mask * 255
        pred_mask_bgr = (cv2.cvtColor(pred_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,0,255], dtype=np.uint8)
        #result = cv2.addWeighted(image, 1, pred_mask_bgr, 0.5, 0)
        #cv2.imwrite(output_images_path + '/' + str(gt_dict['image_id']) + '.jpg', result)
        #exit()
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        # Draw each polygon on the mask
        for polygon in gt_seg_vertices:
            rle = maskUtils.frPyObjects([polygon], h, w)
            m = maskUtils.decode(rle)
            gt_mask = np.maximum(gt_mask, m[:,:,0])
        gt_mask_bool = gt_mask.astype(bool)
        gt_mask_bgr = (cv2.cvtColor(gt_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,255,0], dtype=np.uint8)
        #gt_mask = gt_mask * 255
        #gt_mask_color = cv2.merge([np.zeros_like(gt_mask), gt_mask, np.zeros_like(gt_mask)])

        # Create a binary mask for the outline (1-pixel dilation)
        #kernel = np.ones((3,3), np.uint8)
        #pred_mask_outline = cv2.dilate((pred_mask), kernel, iterations=2)
        #pred_mask_outline = pred_mask_outline - pred_mask
        #pred_mask_outline_color = cv2.merge([np.zeros_like(pred_mask), np.zeros_like(pred_mask), pred_mask_outline])
        #gt_mask_outline = cv2.dilate((gt_mask), kernel, iterations=2)
        #gt_mask_outline = gt_mask_outline - gt_mask
        #gt_mask_outline_color = cv2.merge([np.zeros_like(gt_mask_outline), gt_mask_outline, np.zeros_like(gt_mask_outline)])
        #gt_mask_outline_color = cv2.merge([np.zeros_like(gt_mask), gt_mask_outline, np.zeros_like(gt_mask)])

        #add everything together
        #result = cv2.addWeighted(image, 1, cv2.merge([np.zeros_like(gt_mask), gt_mask, np.zeros_like(gt_mask)]), 0.2, 0)
        result = cv2.addWeighted(image, 1, gt_mask_bgr, 0.5, 0)
        #result = cv2.addWeighted(result, 1, pred_mask_color, 0.2, 0)
        result = cv2.addWeighted(result, 1, pred_mask_bgr, 0.5, 0)
        
        # point1(x,y) = (pr_x,pr_y) is format of writing in bbox detections
        # calculate bounding boxes
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
        bbox_intersection_area = (i_x1 - i_x0) * (i_y1 - i_y0)

        #draw bounding boxes to image
        bbox_image = np.zeros_like(image)
        cv2.rectangle(bbox_image, (pr_x, pr_y), (pr_x + pr_w, pr_y + pr_h), (0,0,255),2)
        cv2.rectangle(bbox_image, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0,255,0),2)
        result = cv2.addWeighted(result, 1, bbox_image, 0.8, 0)

        # write result to video or image file
        if VIDEO:
            out.write(result)
        else:
            cv2.imwrite(output_images_path + '/' + str(gt_dict['image_id']) + '_' + str(gt_dict['category_id']) + '.jpg', result)

        # calculate bbox iou
        bbox_union_area = gt_area + pr_area - bbox_intersection_area
        bbox_iou = bbox_intersection_area/bbox_union_area
        iou_bbox_total += bbox_iou
        print("BBOX IOU: " + str(round(bbox_iou,2)))
        bboxes_found = bboxes_found + 1

        # calculate segmentation mask iou
        intersection = np.logical_and(pred_mask, gt_mask)
        union = np.logical_or(pred_mask, gt_mask)
        pixel_iou = np.sum(intersection) / np.sum(union)
        print("Segmentation IOU: " + str(round(pixel_iou,2)), flush=True)
        iou_seg_total += pixel_iou
        seg_found = seg_found + 1
    ratio_bboxes_found = round(bboxes_found/total_dets, 2)
    iou_bbox = round(iou_bbox_total/bboxes_found, 2)
    ratio_seg_found = round(seg_found/total_dets, 2)
    iou_segs = round(iou_seg_total/seg_found, 2)
    if VIDEO:
        out.release()
    cv2.destroyAllWindows()
    print('Percentage of bounding boxes detected: ' + str(ratio_bboxes_found))
    print('Average Intersection over union for bounding boxes: ' + str(iou_bbox))
    print('Percentage of segmentations detected: ' + str(ratio_seg_found))
    print('Average Intersection over union for segmentations: ' + str(iou_segs))
    f = open(output + '/IOU_results.txt', 'w')
    f.write('Percentage of bounding boxes detected: ' + str(ratio_bboxes_found) + '\n')
    f.write('Average Intersection over union for bounding boxes: ' + str(iou_bbox) + '\n')
    f.write('Percentage of segmentations detected: ' + str(ratio_seg_found) + '\n')
    f.write('Average Intersection over union for segmentations: ' + str(iou_segs) + '\n')
    f.close()

    print("OK!")
        