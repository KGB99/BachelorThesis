import cv2
import os 
import json 
import numpy as np
import pandas as pd
import ast
import pycocotools.mask as maskUtils
from matplotlib import pyplot as plt
import argparse

#pixel accuracy is: (nr. correctly classified pixels) / (total number of pixels)
def calc_pixel_acc(pred_mask_bool, pred_bbox, gt_mask_bool, gt_bbox):
    h,w,_ = pred_mask_bool.shape
    total_pixels = float(h * w)

    mask_correct_pixels = float(np.sum(np.logical_and(pred_mask_bool, gt_mask_bool)))

    #bbox_correct_pixells = # maybe make bool masks before, could help with iou calc too
    pr_x = int(pred_bbox[0])
    pr_y = int(pred_bbox[1])
    pr_w = int(pred_bbox[2])
    pr_h = int(pred_bbox[3])
    gt_x = int(gt_bbox[0])
    gt_y = int(gt_bbox[1])
    gt_w = int(gt_bbox[2])
    gt_h = int(gt_bbox[3])
    i_x0 = max(pr_x, gt_x)
    i_y0 = max(pr_y, gt_y)
    i_x1 = min(pr_x+pr_h, gt_x + gt_h)
    i_y1 = min(pr_y + pr_w, gt_x + gt_w)
    bbox_correct_pixels = float((i_x1 - i_x0) * (i_y1 - i_y0))

    mask_pixel_acc = mask_correct_pixels / total_pixels
    bbox_pixel_acc = bbox_correct_pixels / total_pixels
    return bbox_pixel_acc, mask_pixel_acc

def calc_iou(pred_mask_bool, pred_bbox, gt_mask_bool, gt_bbox):
    # calculate bounding box iou
    pr_x = int(pred_bbox[0])
    pr_y = int(pred_bbox[1])
    pr_w = int(pred_bbox[2])
    pr_h = int(pred_bbox[3])
    pr_area = pr_w * pr_h

    gt_x = int(gt_bbox[0])
    gt_y = int(gt_bbox[1])
    gt_w = int(gt_bbox[2])
    gt_h = int(gt_bbox[3])
    gt_area = gt_w * gt_h

    i_x0 = max(pr_x, gt_x)
    i_y0 = max(pr_y, gt_y)
    i_x1 = min(pr_x+pr_h, gt_x + gt_h)
    i_y1 = min(pr_y + pr_w, gt_x + gt_w)
    
    bbox_intersection_area = (i_x1 - i_x0) * (i_y1 - i_y0)
    bbox_union_area = gt_area + pr_area - bbox_intersection_area
    bbox_iou = bbox_intersection_area/bbox_union_area

    # calculate segmentation mask iou
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    mask_iou = np.sum(intersection) / np.sum(union)
    return bbox_iou, mask_iou

def calc_dice(pred_mask_bool, pred_bbox, gt_mask_bool, gt_bbox):
    #calc and masks
    bbox_and = np.logical_and(pred_bbox, gt_bbox)
    mask_and = np.logical_and(pred_bbox, gt_bbox)
    
    #calc dice coefficients
    bbox_dice = (np.sum(bbox_and) * 2.0) / (np.sum(pred_bbox) + np.sum(gt_bbox))
    mask_dice = (np.sum(mask_and) * 2.0) / (np.sum(pred_mask) + np.sum(gt_mask))

    return bbox_dice, mask_dice

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--preds_path", required=True, type=str)
    parser.add_argument("--bbox_preds", required=False, default='bbox_detections.json', type=str)
    parser.add_argument("--mask_preds", required=False, default='mask_detections.json', type=str)
    parser.add_argument("--labels_path", required=True, type=str)
    parser.add_argument("--images_dir", required=True, type=str)
    parser.add_argument("--save_images", required=False, default=False, type=bool)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--video", required=False, type=bool, default=False)
    args = parser.parse_args()
    images_dir_path = args.images_dir
    #/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/results
    bbox_preds_path = args.preds_path + '/' + args.bbox_preds
    mask_preds_path = args.preds_path + '/' + args.mask_preds
    VIDEO = args.video
    save_images = args.save_images
    #'/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/test_annotations.json'
    test_annotations_path = args.labels_path
    #'/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/predicted_video.mp4'
    output = './results_eval/' + args.output
    if not os.path.isdir(output):
        os.mkdir(output)
    if VIDEO:
        video_path = output + '/predicted_video.mp4'
    if save_images:
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

    # variables for total gts and found predictions
    total_dets_screwdriver = 0
    total_dets_powerdrill = 0
    found_dets_screwdriver = 0
    found_dets_powerdrill = 0
    iteration = 0
    eval_list = []
    for i,gt_dict in enumerate(test_annotations_dict['annotations']):
        # used to store all results for later processing, im using dicts cause appending to df every iteration is too expensive
        eval_dict = {}
        eval_dict['id'] = iteration
        eval_dict['image_id'] = gt_dict['image_id']
        eval_dict['category_id'] = gt_dict['category_id']
        eval_dict['image_file'] = img_mappings[gt_dict['image_id']]

        # load image
        print("Processing Image: " + img_mappings[gt_dict['image_id']])
        gt_bbox = gt_dict['bbox']
        gt_seg_vertices = gt_dict['segmentation']
        gt_img_id = gt_dict['image_id']
        gt_cat_id = gt_dict['category_id']

        #increase according category total
        if (gt_cat_id == 1):
            total_dets_powerdrill += 1
        elif (gt_cat_id == 2):
            total_dets_screwdriver += 1
        
        # load image
        img_path = img_mappings[gt_dict['image_id']]
        image = cv2.imread(img_path)
        h,w,_ = image.shape
        

        # load predicted labels
        try:
            # this makes sure its the same category, so if its not the same category then this does not find something
            pred_bbox = bbox_dict[gt_img_id][gt_cat_id]['bbox']
            pred_seg = mask_dict[gt_img_id][gt_cat_id]['segmentation']
        except KeyError:
            print('Bounding Box or Segmentation not found for this category!')
            eval_dict['prediction_found'] = 0
            eval_dict['bbox_pixel_accuracy'] = 0
            eval_dict['mask_pixel_accuracy'] = 0
            eval_dict['bbox_iou'] = 0
            eval_dict['mask_iou'] = 0
            eval_dict['bbox_dice_coefficient'] = 0
            eval_dict['mask_dice_coefficient'] = 0
            continue
        
        #increase categories found which matches
        if (gt_cat_id == 1):
            found_dets_powerdrill += 1
        elif (gt_cat_id == 2):
            found_dets_screwdriver += 1
        #add to eval_dict that a prediciton was found
        eval_dict['prediction_found'] = 1

        # decode masks
        pred_mask = maskUtils.decode(pred_seg)
        pred_mask_bool = pred_mask.astype(bool)
        #pred_mask = pred_mask * 255
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
        #gt_mask = gt_mask * 255
        #gt_mask_color = cv2.merge([np.zeros_like(gt_mask), gt_mask, np.zeros_like(gt_mask)])
        #gt_mask_rgb = (cv2.cvtColor(gt_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,255,0], dtype=np.uint8)
        #and_mask_rgb = (cv2.cvtColor(and_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([255,0,0], dtype=np.uint8)
        #result_mask_rgb = (cv2.cvtColor(result_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,0,255], dtype=np.uint8)
        # Create a binary mask for the outline (1-pixel dilation)
        #kernel = np.ones((3,3), np.uint8)
        #pred_mask_outline = cv2.dilate((pred_mask), kernel, iterations=2)
        #pred_mask_outline = pred_mask_outline - pred_mask
        #pred_mask_outline_color = cv2.merge([np.zeros_like(pred_mask), np.zeros_like(pred_mask), pred_mask_outline])
        #gt_mask_outline = cv2.dilate((gt_mask), kernel, iterations=2)
        #gt_mask_outline = gt_mask_outline - gt_mask
        #gt_mask_outline_color = cv2.merge([np.zeros_like(gt_mask_outline), gt_mask_outline, np.zeros_like(gt_mask_outline)])
        #gt_mask_outline_color = cv2.merge([np.zeros_like(gt_mask), gt_mask_outline, np.zeros_like(gt_mask)])

        # calculate all relevant evaluation metrics
        bbox_pixel_accuracy, mask_pixel_accuracy = calc_pixel_acc(pred_mask_bool, pred_bbox, gt_mask_bool, gt_bbox)
        bbox_iou, mask_iou = calc_iou(pred_mask_bool, pred_bbox, gt_mask_bool, gt_bbox)
        bbox_dice_coefficient, mask_dice_coefficient = calc_dice(pred_mask_bool, pred_bbox, gt_mask_bool, gt_bbox)
        
        # add values to eval_dict
        eval_dict['bbox_pixel_accuracy'] = bbox_pixel_accuracy
        eval_dict['mask_pixel_accuracy'] = mask_pixel_accuracy
        eval_dict['bbox_iou'] = bbox_iou
        eval_dict['mask_iou'] = mask_iou
        eval_dict['bbox_dice_coefficient'] = bbox_dice_coefficient
        eval_dict['mask_dice_coefficient'] = mask_dice_coefficient
        eval_dict['result_file'] = (output_images_path + '/' + str(gt_dict['image_id']) + '_' + str(gt_dict['category_id']) + '.jpg')
        eval_list.append(eval_dict)
        
        # create a blue mask for the overlapping regions, red for preds and green for ground truth
        and_mask_bool = np.logical_and(pred_mask_bool, gt_mask_bool)
        and_mask_bgr = (cv2.cvtColor(and_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([255,0,0], dtype=np.uint8)
        
        pred_mask_bool = np.logical_and(pred_mask_bool, np.logical_not(and_mask_bool))
        pred_mask_bgr = (cv2.cvtColor(pred_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,0,255], dtype=np.uint8)
        
        gt_mask_bool = np.logical_and(gt_mask_bool, np.logical_not(and_mask_bool))
        gt_mask_bgr = (cv2.cvtColor(gt_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,255,0], dtype=np.uint8)
        #add everything together
        #result = cv2.addWeighted(image, 1, cv2.merge([np.zeros_like(gt_mask), gt_mask, np.zeros_like(gt_mask)]), 0.2, 0)
        result = cv2.addWeighted(image, 1, gt_mask_bgr, 1, 0)
        result = cv2.addWeighted(result, 1, and_mask_bgr, 1, 0)
        result = cv2.addWeighted(result, 1, pred_mask_bgr, 1, 0)


        #draw bounding boxes to image
        #bbox_image = np.zeros_like(image)
        #cv2.rectangle(bbox_image, (pr_x, pr_y), (pr_x + pr_w, pr_y + pr_h), (0,0,255),2)
        #cv2.rectangle(bbox_image, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0,255,0),2)
        #result = cv2.addWeighted(result, 1, bbox_image, 0.8, 0)

        # write result to video or image file
        if VIDEO:
            out.write(result)
        else:
            cv2.imwrite(output_images_path + '/' + str(gt_dict['image_id']) + '_' + str(gt_dict['category_id']) + '.jpg', result)

        # calculate bbox iou
        
        iou_bbox_total += bbox_iou
        print("BBOX IOU: " + str(round(bbox_iou,2)))
        bboxes_found = bboxes_found + 1

        
        print("Segmentation IOU: " + str(round(mask_iou,2)), flush=True)
        iou_seg_total += mask_iou
        seg_found = seg_found + 1
    ratio_bboxes_found = 0#round(bboxes_found/total_dets, 2)
    iou_bbox = round(iou_bbox_total/bboxes_found, 2)
    ratio_seg_found = 0#round(seg_found/total_dets, 2)
    iou_segs = round(iou_seg_total/seg_found, 2)

    # convert eval_list to pandas df and save it as csv for further analysis
    df = pd.DataFrame(eval_list)
    df.to_csv(output + '/eval_values.csv', index=False)

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
        