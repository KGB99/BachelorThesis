import cv2
import os 
import json 
import numpy as np
import pandas as pd
import ast
import pycocotools.mask as maskUtils
from matplotlib import pyplot as plt
import argparse
from pathlib import Path

def eval_yolact(path_to_eval_values):
    df = pd.read_csv(path_to_eval_values)
    p = Path(eval_file_path)
    output = ''
    #print(df.columns.tolist()) : 
    #['pred_powerdrill', 'pred_screwdriver', 'image_file', 'gt_screwdriver', 'pred_screwdriver_conf', 'bbox_iou_screwdriver', 'gt_powerdrill', 'pred_powerdrill_conf', 'bbox_iou_powerdrill']

    threshold_list = list(range(0, 100, 5))
    threshold_list = [x/100 for x in threshold_list]

    # first do everything for bboxes
    eval_dict = {}
    iou_threshold = 0.5
    for conf_threshold in threshold_list:
        
        #calculate confusion matrix for powerdrill bbox
        powerdrill_TP_df = df[(df['pred_powerdrill_conf'] >= conf_threshold) 
                              & (df['bbox_iou_powerdrill'] >= iou_threshold)]
        powerdrill_FP_df = df[(df['pred_powerdrill_conf'] >= conf_threshold) 
                               & (df['bbox_iou_powerdrill'] < iou_threshold)]
        powerdrill_FN_df = df[(df['gt_powerdrill'] == 1) & (df['bbox_iou_powerdrill'] < iou_threshold)]
        
        screwdriver_TP_df = df[(df['pred_screwdriver_conf'] >= conf_threshold) 
                               & (df['bbox_iou_screwdriver'] >= iou_threshold)]
        screwdriver_FP_df = df[((df['pred_screwdriver_conf'] >= conf_threshold) 
                                & (df['bbox_iou_screwdriver'] < iou_threshold))]
        screwdriver_FN_df = df[(df['gt_screwdriver'] == 1) 
                                & (df['bbox_iou_screwdriver'] < iou_threshold)]
        
        eval_dict[conf_threshold] = {}

        eval_dict[conf_threshold]['powerdrill'] = {}
        eval_dict[conf_threshold]['powerdrill']['TP'] = len(powerdrill_TP_df)
        eval_dict[conf_threshold]['powerdrill']['FP'] = len(powerdrill_FP_df)
        eval_dict[conf_threshold]['powerdrill']['FN'] = len(powerdrill_FN_df)
        TP = len(powerdrill_TP_df)
        FP = len(powerdrill_FP_df)
        FN = len(powerdrill_FN_df)
        eval_dict[conf_threshold]['powerdrill']['precision'] = 0 if (TP + FP == 0) else len(powerdrill_TP_df) / (len(powerdrill_TP_df) + len(powerdrill_FP_df))
        eval_dict[conf_threshold]['powerdrill']['recall'] = 0 if (TP + FN == 0) else len(powerdrill_TP_df) / (len(powerdrill_TP_df) + len(powerdrill_FN_df))

        eval_dict[conf_threshold]['screwdriver'] = {}
        eval_dict[conf_threshold]['screwdriver']['TP'] = len(screwdriver_TP_df)
        eval_dict[conf_threshold]['screwdriver']['FP'] = len(screwdriver_FP_df)
        eval_dict[conf_threshold]['screwdriver']['FN'] = len(screwdriver_FN_df)
        TP = len(screwdriver_TP_df)
        FP = len(screwdriver_FP_df)
        FN = len(screwdriver_FN_df)
        eval_dict[conf_threshold]['screwdriver']['precision'] = 0 if (TP + FP == 0) else len(screwdriver_TP_df) / (len(screwdriver_TP_df) + len(screwdriver_FP_df))
        eval_dict[conf_threshold]['screwdriver']['recall'] = 0 if (TP + FN == 0) else len(screwdriver_TP_df) / (len(screwdriver_TP_df) + len(screwdriver_FN_df))
        
        eval_dict[conf_threshold]['total'] = {}
        eval_dict[conf_threshold]['total']['precision'] = (eval_dict[conf_threshold]['screwdriver']['precision'] + eval_dict[conf_threshold]['powerdrill']['precision']) / 2
        eval_dict[conf_threshold]['total']['recall'] = (eval_dict[conf_threshold]['screwdriver']['recall'] + eval_dict[conf_threshold]['powerdrill']['recall']) / 2
        output += (f"Threshold: {conf_threshold:.2f} | Recall: {round(eval_dict[conf_threshold]['total']['recall'], 2):.2f} | Precision: {round(eval_dict[conf_threshold]['total']['precision'], 2):.2f} "
            f"| TP: {len(powerdrill_TP_df) + len(screwdriver_TP_df):4} | FN: {len(powerdrill_FN_df) + len(screwdriver_FN_df):4} | FP: {len(powerdrill_FP_df) + len(screwdriver_FP_df):4} \n")
        

    #prec-rec for total
    total_precision_list = []
    total_recall_list = []
    pow_recall_list = []
    pow_precision_list = []
    screw_recall_list = []
    screw_precision_list = []
    for conf_threshold in eval_dict:
        total_recall_list.append(eval_dict[conf_threshold]['total']['recall'])
        total_precision_list.append(eval_dict[conf_threshold]['total']['precision'])
        pow_recall_list.append(eval_dict[conf_threshold]['powerdrill']['recall'])
        pow_precision_list.append(eval_dict[conf_threshold]['powerdrill']['precision'])
        screw_recall_list.append(eval_dict[conf_threshold]['screwdriver']['recall'])
        screw_precision_list.append(eval_dict[conf_threshold]['screwdriver']['precision'])



    fig, ax = plt.subplots()
    ax.plot(total_recall_list, total_precision_list,marker='o', linestyle='-')
    ax.set_title("Total Precision-Recall Curve : YOLO-" + str(p.parent.name))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True)
    fig.savefig(str(p.parent) + '/' + str(p.parent.name) + '_total-prec-rec.png', dpi=300)

    fig, ax = plt.subplots()
    ax.plot(pow_recall_list, pow_precision_list, marker='o', linestyle='-')
    ax.set_title("Powerdrill Precision-Recall Curve : YOLO-" + str(p.parent.name))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True)
    fig.savefig(str(p.parent) + '/' + str(p.parent.name) + '_power-prec-rec.png', dpi=300)

    fig,ax = plt.subplots()
    ax.plot(screw_recall_list, screw_precision_list, marker='o', linestyle='-')
    ax.set_title("Screwdriver Precision-Recall Curve : YOLO-" + str(p.parent.name))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True)
    fig.savefig(str(p.parent) + '/' + str(p.parent.name) + '_screw-prec-rec.png', dpi=300)
    return output

def eval_yolo(path_to_eval_values):
    df = pd.read_csv(path_to_eval_values)
    p = Path(eval_file_path)
    output = ''
    #print(df.columns.tolist()) : 
    #['pred_powerdrill', 'pred_screwdriver', 'image_file', 'gt_screwdriver', 'pred_screwdriver_conf', 'bbox_iou_screwdriver', 'gt_powerdrill', 'pred_powerdrill_conf', 'bbox_iou_powerdrill']

    threshold_list = list(range(0, 100, 5))
    threshold_list = [x/100 for x in threshold_list]

    # first do everything for bboxes
    eval_dict = {}
    iou_threshold = 0.5
    for conf_threshold in threshold_list:
        
        #calculate confusion matrix for powerdrill bbox
        powerdrill_TP_df = df[(df['pred_powerdrill_conf'] >= conf_threshold) 
                              & (df['bbox_iou_powerdrill'] >= iou_threshold)]
        powerdrill_FP_df = df[(df['pred_powerdrill_conf'] >= conf_threshold) 
                               & (df['bbox_iou_powerdrill'] < iou_threshold)]
        #powerdrill_FN_df = df[(df['gt_powerdrill'] == 1) & (df['bbox_iou_powerdrill'] < iou_threshold)]
        powerdrill_FN_df = df[(df['gt_powerdrill'] == 1) & (df['pred_powerdrill'] == 1) & (df['pred_screwdriver'] == 1)]
        
        screwdriver_TP_df = df[(df['pred_screwdriver_conf'] >= conf_threshold) 
                               & (df['bbox_iou_screwdriver'] >= iou_threshold)]
        screwdriver_FP_df = df[((df['pred_screwdriver_conf'] >= conf_threshold) 
                                & (df['bbox_iou_screwdriver'] < iou_threshold))]
        screwdriver_FN_df = df[(df['gt_screwdriver'] == 1) 
                                & (df['bbox_iou_screwdriver'] < iou_threshold)]
        
        eval_dict[conf_threshold] = {}

        eval_dict[conf_threshold]['powerdrill'] = {}
        eval_dict[conf_threshold]['powerdrill']['TP'] = len(powerdrill_TP_df)
        eval_dict[conf_threshold]['powerdrill']['FP'] = len(powerdrill_FP_df) 
        eval_dict[conf_threshold]['powerdrill']['FN'] = len(powerdrill_FN_df)
        eval_dict[conf_threshold]['powerdrill']['precision'] = len(powerdrill_TP_df) / (len(powerdrill_TP_df) + len(powerdrill_FP_df))
        eval_dict[conf_threshold]['powerdrill']['recall'] = len(powerdrill_TP_df) / (len(powerdrill_TP_df) + len(powerdrill_FN_df))

        eval_dict[conf_threshold]['screwdriver'] = {}
        eval_dict[conf_threshold]['screwdriver']['TP'] = len(screwdriver_TP_df)
        eval_dict[conf_threshold]['screwdriver']['FP'] = len(screwdriver_FP_df)
        eval_dict[conf_threshold]['screwdriver']['FN'] = len(screwdriver_FN_df)
        eval_dict[conf_threshold]['screwdriver']['precision'] = len(screwdriver_TP_df) / (len(screwdriver_TP_df) + len(screwdriver_FP_df))
        eval_dict[conf_threshold]['screwdriver']['recall'] = len(screwdriver_TP_df) / (len(screwdriver_TP_df) + len(screwdriver_FN_df))
        
        eval_dict[conf_threshold]['total'] = {}
        eval_dict[conf_threshold]['total']['precision'] = (eval_dict[conf_threshold]['screwdriver']['precision'] + eval_dict[conf_threshold]['powerdrill']['precision']) / 2
        eval_dict[conf_threshold]['total']['recall'] = (eval_dict[conf_threshold]['screwdriver']['recall'] + eval_dict[conf_threshold]['powerdrill']['recall']) / 2

        output += (f"Threshold: {conf_threshold:.2f} | Recall: {round(eval_dict[conf_threshold]['total']['recall'], 2):.2f} | Precision: {round(eval_dict[conf_threshold]['total']['precision'], 2):.2f} "
            f"| TP: {len(powerdrill_TP_df) + len(screwdriver_TP_df):4} | FN: {len(powerdrill_FN_df) + len(screwdriver_FN_df):4} | FP: {len(powerdrill_FP_df) + len(screwdriver_FP_df):4} \n")
        

    #prec-rec for total
    total_precision_list = []
    total_recall_list = []
    pow_recall_list = []
    pow_precision_list = []
    screw_recall_list = []
    screw_precision_list = []
    for conf_threshold in eval_dict:
        total_recall_list.append(eval_dict[conf_threshold]['total']['recall'])
        total_precision_list.append(eval_dict[conf_threshold]['total']['precision'])
        pow_recall_list.append(eval_dict[conf_threshold]['powerdrill']['recall'])
        pow_precision_list.append(eval_dict[conf_threshold]['powerdrill']['precision'])
        screw_recall_list.append(eval_dict[conf_threshold]['screwdriver']['recall'])
        screw_precision_list.append(eval_dict[conf_threshold]['screwdriver']['precision'])



    fig, ax = plt.subplots()
    ax.plot(total_recall_list, total_precision_list,marker='o', linestyle='-')
    ax.set_title("Total Precision-Recall Curve : YOLO-" + str(p.parent.name))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True)
    fig.savefig(str(p.parent) + '/' + str(p.parent.name) + '_total-prec-rec.png', dpi=300)

    fig, ax = plt.subplots()
    ax.plot(pow_recall_list, pow_precision_list, marker='o', linestyle='-')
    ax.set_title("Powerdrill Precision-Recall Curve : YOLO-" + str(p.parent.name))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True)
    fig.savefig(str(p.parent) + '/' + str(p.parent.name) + '_power-prec-rec.png', dpi=300)

    fig,ax = plt.subplots()
    ax.plot(screw_recall_list, screw_precision_list, marker='o', linestyle='-')
    ax.set_title("Screwdriver Precision-Recall Curve : YOLO-" + str(p.parent.name))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True)
    fig.savefig(str(p.parent) + '/' + str(p.parent.name) + '_screw-prec-rec.png', dpi=300)
    
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--yolo", required=False ,default=False, type=bool)
    parser.add_argument("--yolact", required=False,default=False, type=bool)
    args = parser.parse_args()
    if args.yolo:
        parent_path = '/Users/kerim/dev/BachelorThesis/results_eval/YOLO'
        yolo_folders = os.listdir(parent_path)
        for folder in yolo_folders:
            eval_file_path = parent_path + '/' + folder + '/eval_values.csv'
            output = eval_yolo(eval_file_path)
            with open(parent_path + '/' + folder + '/eval_output.txt', 'w') as f:
                f.write(output)
            print(output)
    if args.yolact:
        parent_path = '/Users/kerim/dev/BachelorThesis/results_eval/YOLACT'
        yolact_folders = os.listdir(parent_path)
        for folder in yolact_folders:
            eval_file_path = parent_path + '/' + folder + '/eval_values.csv'
            output = eval_yolact(eval_file_path)
            with open(parent_path + '/' + folder + '/eval_output.txt', 'w') as f:
                f.write(output)
            print(output)
    print("OK!")