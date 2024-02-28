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
def calc_pixel_acc(pred_mask_bool, gt_mask_bool):
    h,w = pred_mask_bool.shape
    total_pixels = float(h * w)

    mask_correct_pixels = float(np.sum(np.logical_and(pred_mask_bool, gt_mask_bool)))

    #bbox_correct_pixells = # maybe make bool masks before, could help with iou calc too

    # only calc if (total_pixels > 0) otherwise mask_pixel_acc should be 0
    mask_pixel_acc = (mask_correct_pixels / total_pixels)
    #bbox_pixel_acc = bbox_correct_pixels / total_pixels
    return mask_pixel_acc

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
    i_x1 = min(pr_x+pr_w, gt_x + gt_w)
    i_y1 = min(pr_y + pr_h, gt_y + gt_h)
    
    bbox_intersection_area = max(0,(i_x1 - i_x0)) * max(0,(i_y1 - i_y0)) # if either value is negative then area is 0
    bbox_union_area = gt_area + pr_area - bbox_intersection_area
    bbox_iou = bbox_intersection_area/bbox_union_area

    # calculate segmentation mask iou
    intersection = np.logical_and(pred_mask_bool, gt_mask_bool)
    union = np.logical_or(pred_mask_bool, gt_mask_bool)
    mask_iou = np.sum(intersection) / np.sum(union)
    return bbox_iou, mask_iou

def calc_dice(pred_mask_bool, gt_mask_bool):
    #calc and masks
    #bbox_and = np.logical_and(pred_bbox, gt_bbox)
    mask_and = np.logical_and(pred_mask_bool, gt_mask_bool)
    
    #calc dice coefficients
    #bbox_dice = (np.sum(bbox_and) * 2.0) / (np.sum(pred_bbox) + np.sum(gt_bbox))
    mask_dice = (np.sum(mask_and) * 2.0) / (np.sum(pred_mask_bool) + np.sum(gt_mask_bool))

    return mask_dice#bbox_dice, mask_dice

def eval_yolo(args):
    test_annotations_path = args.labels_path
    images_dir_path = args.images_dir
    output = './results_eval/' + args.output
    preds_path = args.preds_path # should be the folder with all the .txt
    
    if not os.path.isdir(output):
        os.mkdir(output)

    if args.save_images:
        if not os.path.isdir(output + '/evaluatedImages'):
            os.mkdir(output + '/evaluatedImages')
        output_images_path = output + '/evaluatedImages'

    f = open(test_annotations_path, 'r')
    test_annotations_dict = json.load(f)
    f.close()

    # dictionary for all evaluations
    eval_dict = {}

    # get all the files in the preds directory and iterate through them while saving the predictions in a dict
    preds_path_files = os.listdir(preds_path)
    preds_dict = {}
    for preds_file in preds_path_files:
        # due to my naming convention first 6 letters are cam id and following 6 are img id
        camera_id = preds_file[:6]
        image_id = preds_file[6:12] 
        if not (camera_id in preds_dict):
            preds_dict[camera_id] = {}
        if not (image_id in preds_dict[camera_id]):
            preds_dict[camera_id][image_id] = {}
            preds_dict[camera_id][image_id]['
            with open(preds_path + '/' + preds_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line[:-1] # this makes sure /n does not come with
                    
                exit()
        
    return

def eval_yolact(args):
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
    
    # dictionary for all evaluations
    eval_dict = {}
    
    # read bbox predictions, only keep the predictions with highest scores
    bbox_dict = {}
    f = open(bbox_preds_path, 'r')
    line_list = ast.literal_eval(f.readline())
    for line_dict in line_list:
        curr_id = line_dict['image_id']
        curr_cat = line_dict['category_id']
        
        #note down that the prediction exists
        if curr_id not in eval_dict:
            eval_dict[curr_id] = {}
            eval_dict[curr_id]['pred_exists'] = 1
        
        if curr_id in bbox_dict:
            if curr_cat in bbox_dict[curr_id]:
                if (line_dict['score'] > (bbox_dict[curr_id][curr_cat]['score'])):
                    bbox_dict[curr_id][curr_cat]['bbox'] = line_dict['bbox']
                    bbox_dict[curr_id][curr_cat]['score'] = line_dict['score']
            else:
                bbox_dict[curr_id][curr_cat] = {}
                bbox_dict[curr_id][curr_cat]['bbox'] = line_dict['bbox']
                bbox_dict[curr_id][curr_cat]['score'] = line_dict['score']
        else:
            bbox_dict[curr_id] = {}
            bbox_dict[curr_id][curr_cat] = {}
            bbox_dict[curr_id][curr_cat]['bbox'] = line_dict['bbox']
            bbox_dict[curr_id][curr_cat]['score'] = line_dict['score']
    f.close()

    # read mask predictions, only keep the predictions with highest scores
    mask_dict = {}
    f = open(mask_preds_path, 'r')
    line_list = ast.literal_eval(f.readline())
    for line_dict in line_list:
        curr_id = line_dict['image_id']
        curr_cat = line_dict['category_id']
        
        if curr_id not in eval_dict:
            eval_dict[curr_id] = {}
            eval_dict[curr_id]['pred_exists'] = 1
            
        if curr_id in mask_dict:
            if curr_cat in mask_dict[curr_id]:
                if line_dict['score'] > mask_dict[curr_id][curr_cat]['score']:
                    mask_dict[curr_id][curr_cat]['segmentation' ] = line_dict['segmentation']
                    mask_dict[curr_id][curr_cat]['score'] = line_dict['score']
            else:
                mask_dict[curr_id][curr_cat] = {}
                mask_dict[curr_id][curr_cat]['segmentation' ] = line_dict['segmentation']
                mask_dict[curr_id][curr_cat]['score'] = line_dict['score']
        else:
            mask_dict[curr_id] = {}
            mask_dict[curr_id][curr_cat] = {}
            mask_dict[curr_id][curr_cat]['segmentation' ] = line_dict['segmentation']
            mask_dict[curr_id][curr_cat]['score'] = line_dict['score']
    f.close()
    
    # create mapping for img_id -> img_path
    #img_mappings = {}
    #path_prepend = images_dir_path + '/'
    #for camera in test_annotations_dict:
    #    for image in test_annotations_dict[camera]:
    #        img_dict = test_annotations_dict[camera][image]['img']
    #        img_mappings[img_dict['id']] = path_prepend + img_dict['file_name']


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
        
    # value for status update
    len_test_annotations_dict = len(test_annotations_dict)
    
    # variables for total gts and found predictions
    total_dets_screwdriver = 0
    total_dets_powerdrill = 0
    found_dets_screwdriver = 0
    found_dets_powerdrill = 0
    iteration = 0
    for i,camera in enumerate(test_annotations_dict):
        len_camera_dict = len(test_annotations_dict[camera])
        for j, image in enumerate(test_annotations_dict[camera]):
            # load the gt of the image at this camera
            gt_dict = test_annotations_dict[camera][image]
            curr_id = gt_dict['img']['id'] # this is the unique id i provided in makeCoco.py so each id is unique to an image

            # status update print
            print(f"Camera: {i:02} / {len_test_annotations_dict} "
                  f"| Image: {j:05} / {len_camera_dict} "
                  f"| Filename: {gt_dict['img']['file_name']} ", 
                  flush=True)
                  
            # used to store all results for later processing, im using dicts cause appending to df every iteration is too expensive
            if curr_id not in eval_dict:
                eval_dict[curr_id] = {}
                # if at this point there is no dictionary yet, that means there is no prediction for it
                eval_dict[curr_id]['pred_powerdrill'] = 0
                eval_dict[curr_id]['pred_screwdriver'] = 0
            #eval_dict['id'] = iteration
            #eval_dict[curr_id]['image_id'] = gt_dict['img']['id']
            eval_dict[curr_id]['image_file'] = gt_dict['img']['file_name']

            if gt_dict['gt_exists'] == 0:
                eval_dict[curr_id]['gt_powerdrill'] = 0
                eval_dict[curr_id]['gt_screwdriver'] = 0
                continue

            for gt_mask_dict in gt_dict['masks']:
                if (gt_mask_dict['category_id'] == 1):
                    eval_dict[curr_id]['gt_powerdrill'] = 1
                    tool = 'powerdrill'
                elif (gt_mask_dict['category_id'] == 2):
                    eval_dict[curr_id]['gt_screwdriver'] = 1
                    tool = 'screwdriver'
                    
                gt_bbox = gt_mask_dict['bbox']
                gt_seg_vertices = gt_mask_dict['segmentation']
                gt_img_id = gt_mask_dict['image_id']
                gt_cat_id = gt_mask_dict['category_id']

                
                
            
                #increase according category total
                if (gt_cat_id == 1):
                    total_dets_powerdrill += 1
                elif (gt_cat_id == 2):
                    total_dets_screwdriver += 1

                # load image
                img_path = images_dir_path + '/' + gt_dict['img']['file_name']
                image = cv2.imread(img_path)
                h,w,_ = image.shape


                # load predicted labels
                try:
                    # this makes sure its the same category, so if its not the same category then this does not find something
                    pred_bbox = bbox_dict[gt_img_id][gt_cat_id]['bbox']
                    pred_conf = bbox_dict[gt_img_id][gt_cat_id]['score']
                    pred_seg = mask_dict[gt_img_id][gt_cat_id]['segmentation']
                    # i believe segmentation and bbox confidence should be the same since its part of the same prediction
                    #pred_seg_conf = mask_dict[gt_img_id][gt_cat_id]['score']
                    #if pred_bbox_conf == pred_seg_conf:
                    #    print("BOTH SAME")
                    #continue
                    
                except KeyError:
                    print('No prediction found for this category!')
                    # i think at this point pred_exists should already be 0, but just incase
                    #eval_dict[curr_id]['pred_exists'] = 0
                    continue
                if gt_cat_id == 1:
                    eval_dict[curr_id]['pred_powerdrill'] = 1
                    eval_dict[curr_id]['pred_powerdrill_conf'] = pred_conf
                elif gt_cat_id == 2:
                    eval_dict[curr_id]['pred_screwdriver'] = 1
                    eval_dict[curr_id]['pred_screwdriver_conf'] = pred_conf
                else:
                    raise ValueError("This gt is neither a powerdrill nor a screwdriver!")

                # decode masks
                pred_mask = maskUtils.decode(pred_seg)
                pred_mask_bool = pred_mask.astype(bool)

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
                #bbox_pixel_accuracy, mask_pixel_accuracy = calc_pixel_acc(pred_mask_bool, pred_bbox, gt_mask_bool, gt_bbox)
                mask_pixel_accuracy = calc_pixel_acc(pred_mask_bool, gt_mask_bool)
                bbox_iou, mask_iou = calc_iou(pred_mask_bool, pred_bbox, gt_mask_bool, gt_bbox)
                #bbox_dice_coefficient, mask_dice_coefficient = calc_dice(pred_mask_bool, pred_bbox, gt_mask_bool, gt_bbox)
                mask_dice_coefficient = calc_dice(pred_mask_bool, gt_mask_bool)

                eval_dict[curr_id]['mask_pixel_accuracy_' + tool] = mask_pixel_accuracy
                eval_dict[curr_id]['bbox_iou_' + tool] = bbox_iou
                eval_dict[curr_id]['mask_iou_' + tool] = mask_iou
                #eval_dict['bbox_dice_coefficient'] = bbox_dice_coefficient
                eval_dict[curr_id]['mask_dice_coefficient_' + tool] = mask_dice_coefficient
                eval_dict[curr_id]['result_img_file_' + tool] = (output_images_path + '/' + str(gt_dict['img']['id']) + '_' \
                                                + str(gt_mask_dict['category_id']) + '.jpg')
                if save_images:
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
                    pr_x = int(pred_bbox[0])
                    pr_y = int(pred_bbox[1])
                    pr_w = int(pred_bbox[2])
                    pr_h = int(pred_bbox[3])
                    gt_x = int(gt_bbox[0])
                    gt_y = int(gt_bbox[1])
                    gt_w = int(gt_bbox[2])
                    gt_h = int(gt_bbox[3])
                    bbox_image = np.zeros_like(image)
                    cv2.rectangle(bbox_image, (pr_x, pr_y), (pr_x + pr_w, pr_y + pr_h), (0,0,255),2)
                    cv2.rectangle(bbox_image, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0,255,0),2)
                    result = cv2.addWeighted(result, 1, bbox_image, 1, 0)

                    # write result to video or image file
                    #if VIDEO:
                    #    out.write(result)
                    cv2.imwrite(output_images_path + '/' + str(gt_dict['img']['id']) + '_' 
                                 + str(gt_mask_dict['category_id']) + '.jpg', result)

                # calculate bbox iou

                iou_bbox_total += bbox_iou
                print("BBOX IOU: " + str(round(bbox_iou,2)))
                bboxes_found = bboxes_found + 1


                print("Segmentation IOU: " + str(round(mask_iou,2)), flush=True)
                iou_seg_total += mask_iou
                seg_found = seg_found + 1
    
    # final few calculations and saving of the processed data
    ratio_bboxes_found = 0#round(bboxes_found/total_dets, 2)
    iou_bbox = round(iou_bbox_total/bboxes_found, 2)
    ratio_seg_found = 0#round(seg_found/total_dets, 2)
    iou_segs = round(iou_seg_total/seg_found, 2)

    # convert eval_list to pandas df and save it as csv for further analysis
    df = pd.DataFrame.from_dict(eval_dict, orient='index')
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
    return

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
    parser.add_argument("--yolact", required=False, default=False, type=bool)
    parser.add_argument("--yolo", required=False, default=False, type=bool)
    args = parser.parse_args()
    
    if args.yolact:
        eval_yolact(args)
    elif args.yolo:
        eval_yolo(args)

    print("OK!")
        
