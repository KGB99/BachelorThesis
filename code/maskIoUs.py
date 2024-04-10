import cv2
import os 
import json 
import numpy as np
import pandas as pd
import ast
import pycocotools.mask as maskUtils
from matplotlib import pyplot as plt
import argparse

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

def calc_vals(gt_labels, gen_labels_path, output_folder):
    images_dir_path = '/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp/'
    
    with open(gen_labels_path, 'r') as f:
        gen_labels = json.load(f)
        
    total_bbox = 0.0
    total_mask = 0.0
    total_iterations = 0
    
    len_gen_labels = len(gen_labels)
    for i,camera in enumerate(gen_labels):
        len_cam = len(gen_labels[camera])
        #if (camera != '016003'):
        #    continue
        for j,imageId in enumerate(gen_labels[camera]):
            print(f"Camera={i+1:2}/{len_gen_labels} | Image={j+1:2}/{len_cam}")
            gen_mask_dict = gen_labels[camera][imageId]['mask']
            gt_mask_dict = gt_labels[camera][imageId]['mask']
            gen_bbox = gen_mask_dict['bbox']
            gen_segs = gen_mask_dict['segmentation']
            gt_bbox = gt_mask_dict['bbox']
            gt_segs = gt_mask_dict['segmentation']

            img_path = images_dir_path + gt_labels[camera][imageId]['img']['file_name']
            image = cv2.imread(img_path)
            h,w,_ = image.shape
            
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            for gt_polygon in gt_segs:
                gt_run_length_encoding = maskUtils.frPyObjects([gt_polygon], h, w)
                masks = maskUtils.decode(gt_run_length_encoding)
                gt_mask = np.maximum(gt_mask, masks[:,:,0])
            gt_mask_bool = gt_mask.astype(bool)
            
            gen_mask = np.zeros((h, w), dtype=np.uint8)
            for gen_polygon in gen_segs:
                gen_run_length_encoding = maskUtils.frPyObjects([gen_polygon], h, w)
                masks = maskUtils.decode(gen_run_length_encoding)
                gen_mask = np.maximum(gen_mask, masks[:,:,0])
            gen_mask_bool = gen_mask.astype(bool)
            

            and_mask_bool = np.logical_and(gen_mask_bool, gt_mask_bool)
            gen_mask_bool = np.logical_and(gen_mask_bool, np.logical_not(and_mask_bool))
            gt_mask_bool = np.logical_and(gt_mask_bool, np.logical_not(and_mask_bool))
            gt_mask_rgb = (cv2.cvtColor(gt_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,255,0], dtype=np.uint8)
            and_mask_rgb = (cv2.cvtColor(and_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([255,0,0], dtype=np.uint8)
            gen_mask_rgb = (cv2.cvtColor(gen_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,0,255], dtype=np.uint8)

            mask_transparency = 0.5
            result_image = cv2.addWeighted(image, 1.0, and_mask_rgb, mask_transparency, 0)
            result_image = cv2.addWeighted(result_image, 1.0, gen_mask_rgb, mask_transparency, 0)
            result_image = cv2.addWeighted(result_image, 1.0, gt_mask_rgb, mask_transparency, 0)

            #draw bounding boxes to image
            pr_x = int(gen_bbox[0])
            pr_y = int(gen_bbox[1])
            pr_w = int(gen_bbox[2])
            pr_h = int(gen_bbox[3])
            gt_x = int(gt_bbox[0])
            gt_y = int(gt_bbox[1])
            gt_w = int(gt_bbox[2])
            gt_h = int(gt_bbox[3])
            bbox_image = np.zeros_like(image)
            cv2.rectangle(bbox_image, (pr_x, pr_y), (pr_x + pr_w, pr_y + pr_h), (0,0,255),2)
            cv2.rectangle(bbox_image, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0,255,0),2)
            result_image = cv2.addWeighted(result_image, 1, bbox_image, 0.8, 0)

            cv2.imwrite(f"/cluster/project/infk/cvg/heinj/students/kbirgi/BachelorThesis/visuals_output/{output_folder}/{camera}_{[imageId]['img']['file_name']}", result_image)
            
            bbox_iou, mask_iou = calc_iou(gen_mask_bool, gen_bbox, gt_mask_bool, gt_bbox)
            
            total_iterations += 1
            total_bbox += bbox_iou
            total_mask += mask_iou
    
    avg_bbox_iou = total_bbox / total_iterations
    avg_mask_iou = total_mask / total_iterations
    
    return avg_bbox_iou, avg_mask_iou

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    #parser.add_argument("--gt_labels_path", required=False, default='/cluster/project/infk/cvg/heinj/students/kbirgi/Annotations/trainSSD/amodal_labels_50.json', type=str)
    parser.add_argument("--gt_labels_path", required=False, default='/cluster/project/infk/cvg/heinj/students/kbirgi/Annotations/trainSSD/amodal_labels.json', type=str)
    args = parser.parse_args()
    
    gen_labels1 = '/cluster/project/infk/cvg/heinj/students/kbirgi/Annotations/gen_annotations/stride_50_pbr_base_30000/generated_labels.json'
    
    gen_labels2 = '/cluster/project/infk/cvg/heinj/students/kbirgi/Annotations/gen_annotations/stride_50_pbr_ref_all_no_noise_27000/generated_labels.json'
    
    print("reading gt...", end='')
    with open(args.gt_labels_path, 'r') as f:
        gt_labels = json.load(f)
    print(" done.", flush=True)
    
    avg_bbox_iou, avg_mask_iou = calc_vals(gt_labels, gen_labels1, "base")
    
    with open('./gen_anns_evals_pbr_base.txt', 'w') as f:
        f.write(f"AVG_BBOX_IOU={round(avg_bbox_iou,4)} | AVG_MASK_IOU={round(avg_mask_iou,4)}")
        
    avg_bbox_iou, avg_mask_iou = calc_vals(gt_labels, gen_labels2, "real")
    
    with open('./gen_anns_evals_ref_real.txt', 'w') as f:
        f.write(f"AVG_BBOX_IOU={avg_bbox_iou:.4} | AVG_MASK_IOU={avg_mask_iou:.4}")
    
    print("OK!")
            
            
            