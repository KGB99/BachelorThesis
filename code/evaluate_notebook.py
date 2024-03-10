import cv2
import json 
import os 
import numpy as np
import pandas as pd
import ast
import pycocotools.mask as maskUtils
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
from sklearn import metrics

def interpolate_prec(precision_list):
    for i in (range(len(precision_list)))[1:]:
        if precision_list[i] < precision_list[i-1]:
            precision_list[i] = precision_list[i-1]
    return precision_list

def eval_yolact(path_to_eval_values, iou_threshold):
    df = pd.read_csv(path_to_eval_values)
    p = Path(eval_file_path)
    output = f"IoU-Threshold: {iou_threshold}\n"
    #print(df.columns.tolist()) : 
    #['pred_powerdrill', 'pred_screwdriver', 'image_file', 'gt_screwdriver', 'pred_screwdriver_conf', 'bbox_iou_screwdriver', 'gt_powerdrill', 'pred_powerdrill_conf', 'bbox_iou_powerdrill']

    threshold_list = np.arange(0, 1.01, 0.01)
    #threshold_list = list(range(0,101, 1))
    #threshold_list = [x/100 for x in threshold_list]

    # first do everything for bboxes
    eval_dict = {}
    for conf_threshold in threshold_list:
        
        #calculate confusion matrix for powerdrill bbox
        powerdrill_TP_df = df[((df['pred_powerdrill_conf'] >= conf_threshold) 
                              & (df['bbox_iou_powerdrill'] >= iou_threshold))]
        powerdrill_FP_df = df[((df['pred_powerdrill_conf'] >= conf_threshold) 
                               & (df['bbox_iou_powerdrill'] < iou_threshold))]
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
        PWD_TP = len(powerdrill_TP_df)
        PWD_FP = len(powerdrill_FP_df)
        PWD_FN = len(powerdrill_FN_df)
        power_undefined = True
        if ((PWD_TP + PWD_FP != 0) & (PWD_TP + PWD_FN != 0)):
            power_undefined = False
            eval_dict[conf_threshold]['powerdrill']['precision'] = len(powerdrill_TP_df) / (len(powerdrill_TP_df) + len(powerdrill_FP_df))
        #if (TP + FN != 0):
            eval_dict[conf_threshold]['powerdrill']['recall'] = len(powerdrill_TP_df) / (len(powerdrill_TP_df) + len(powerdrill_FN_df))

        eval_dict[conf_threshold]['screwdriver'] = {}
        eval_dict[conf_threshold]['screwdriver']['TP'] = len(screwdriver_TP_df)
        eval_dict[conf_threshold]['screwdriver']['FP'] = len(screwdriver_FP_df)
        eval_dict[conf_threshold]['screwdriver']['FN'] = len(screwdriver_FN_df)
        SCR_TP = len(screwdriver_TP_df)
        SCR_FP = len(screwdriver_FP_df)
        SCR_FN = len(screwdriver_FN_df)
        screw_undefined = True
        if ((SCR_TP + SCR_FP != 0) & (SCR_TP + SCR_FN != 0)):
            screw_undefined = False
            eval_dict[conf_threshold]['screwdriver']['precision'] = len(screwdriver_TP_df) / (len(screwdriver_TP_df) + len(screwdriver_FP_df))
        #if (TP + FN != 0):
            eval_dict[conf_threshold]['screwdriver']['recall'] = len(screwdriver_TP_df) / (len(screwdriver_TP_df) + len(screwdriver_FN_df))
        
        if (power_undefined or screw_undefined):
            output += (f"Confidence-Threshold: {conf_threshold:.2f} | Recall: Undefined | Precision: Undefined "
            f"| TP: {len(powerdrill_TP_df) + len(screwdriver_TP_df):4} | FN: {len(powerdrill_FN_df) + len(screwdriver_FN_df):4} | FP: {len(powerdrill_FP_df) + len(screwdriver_FP_df):4} \n")
            continue
        eval_dict[conf_threshold]['total'] = {}
        eval_dict[conf_threshold]['total']['precision'] = (eval_dict[conf_threshold]['screwdriver']['precision'] + eval_dict[conf_threshold]['powerdrill']['precision']) / 2
        eval_dict[conf_threshold]['total']['recall'] = (eval_dict[conf_threshold]['screwdriver']['recall'] + eval_dict[conf_threshold]['powerdrill']['recall']) / 2
        output += (f"Confidence-Threshold: {conf_threshold:.2f} | Recall: {round(eval_dict[conf_threshold]['total']['recall'], 2):.2f} | Precision: {round(eval_dict[conf_threshold]['total']['precision'], 2):.2f} "
            f"| TP: {len(powerdrill_TP_df) + len(screwdriver_TP_df):4} | FN: {len(powerdrill_FN_df) + len(screwdriver_FN_df):4} | FP: {len(powerdrill_FP_df) + len(screwdriver_FP_df):4} \n")
        

    #prec-rec for total
    total_precision_list = []
    total_recall_list = []
    pow_recall_list = []
    pow_precision_list = []
    screw_recall_list = []
    screw_precision_list = []
    for conf_threshold in eval_dict:
        if 'total' in eval_dict[conf_threshold]:
            total_recall_list.append(eval_dict[conf_threshold]['total']['recall'])
            total_precision_list.append(eval_dict[conf_threshold]['total']['precision'])
        if (('recall' in eval_dict[conf_threshold]['powerdrill']) and ('precision' in eval_dict[conf_threshold]['powerdrill'])):
            pow_recall_list.append(eval_dict[conf_threshold]['powerdrill']['recall'])
            pow_precision_list.append(eval_dict[conf_threshold]['powerdrill']['precision'])
        if (('recall' in eval_dict[conf_threshold]['screwdriver']) and ('precision' in eval_dict[conf_threshold]['screwdriver'])):
            screw_recall_list.append(eval_dict[conf_threshold]['screwdriver']['recall'])
            screw_precision_list.append(eval_dict[conf_threshold]['screwdriver']['precision'])

    # the documentation for sklearn.metrics says the lists have to be sorted monotonically at the recall list
    # source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc
    # therefore we sort both lists before calculating area under curve (auc)
    # this is now old stuff i use AP
    #np_pow_recall_list = np.array(pow_recall_list)
    #np_pow_prec_list = np.array(pow_precision_list)
    #sorted_powerdrill = np.argsort(np_pow_recall_list)
    #sorted_pow_recall = np_pow_recall_list[sorted_powerdrill]
    #sorted_pow_prec = np_pow_prec_list[sorted_powerdrill]
    #AUC_powerdrill = metrics.auc(sorted_pow_recall, sorted_pow_prec)

    #np_screw_recall_list = np.array(screw_recall_list)
    #np_screw_prec_list = np.array(screw_precision_list)
    #sorted_screwdriver = np.argsort(np_screw_recall_list)
    #sorted_screw_recall = np_screw_recall_list[sorted_screwdriver]
    #sorted_screw_precision = np_screw_prec_list[sorted_screwdriver]
    #AUC_screwdriver = metrics.auc(sorted_screw_recall, sorted_screw_precision)

    #auc = 0.5 * (AUC_screwdriver + AUC_powerdrill)
    #output += (f"auc: {auc} \n\n\n-------------------------------\n\n\n")
    interpolated_pow_prec = interpolate_prec(pow_precision_list)
    interpolated_screw_prec = interpolate_prec(screw_precision_list)
    interpolated_total_prec = interpolate_prec(total_precision_list)
    if args.plot:
        fig, ax = plt.subplots()
        ax.plot(total_recall_list, total_precision_list,linestyle='-')
        ax.set_title("Total Precision-Recall Curve : YOLO-" + str(p.parent.name))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True, alpha=0.3)
        fig.savefig(str(p.parent) + '/' + str(p.parent.name) + '_total-prec-rec.png', dpi=300)
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(pow_recall_list, pow_precision_list, marker='o', linestyle='-')
        ax.set_title("Powerdrill Precision-Recall Curve : YOLO-" + str(p.parent.name))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True, alpha=0.3)
        fig.savefig(str(p.parent) + '/' + str(p.parent.name) + '_power-prec-rec.png', dpi=300)
        plt.close()

        fig,ax = plt.subplots()
        ax.plot(screw_recall_list, screw_precision_list, marker='o', linestyle='-')
        ax.set_title("Screwdriver Precision-Recall Curve : YOLO-" + str(p.parent.name))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True, alpha=0.3)
        fig.savefig(str(p.parent) + '/' + str(p.parent.name) + '_screw-prec-rec.png', dpi=300)
        plt.close()

    if args.simple_output:
        output = ''
    ap = 0.0
    for precision in interpolated_total_prec:
        ap += precision
    ap = ap / 101
    output += f"AP@[{iou_threshold:.2f}]={round(ap,2)}\n"
    return output,ap

def eval_yolo(path_to_eval_values, iou_threshold, PLOT):
    df = pd.read_csv(path_to_eval_values)
    p = Path(eval_file_path)
    output = f"IoU-Threshold: {iou_threshold}\n"
    #print(df.columns.tolist()) : 
    #['pred_powerdrill', 'pred_screwdriver', 'image_file', 'gt_screwdriver', 'pred_screwdriver_conf', 'bbox_iou_screwdriver', 'gt_powerdrill', 'pred_powerdrill_conf', 'bbox_iou_powerdrill']

    #threshold_list = list(range(0, 100, 5))
    #threshold_list = [x/100 for x in threshold_list]
    threshold_list = np.arange(0, 1.01, 0.01)

    # first do everything for bboxes
    eval_dict = {}
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
        PWD_TP = len(powerdrill_TP_df)
        PWD_FP = len(powerdrill_FP_df)
        PWD_FN = len(powerdrill_FN_df)
        power_undefined = True
        if ((PWD_TP + PWD_FP != 0) & (PWD_TP + PWD_FN != 0)):
            power_undefined = False
            eval_dict[conf_threshold]['powerdrill']['precision'] = len(powerdrill_TP_df) / (len(powerdrill_TP_df) + len(powerdrill_FP_df))
        #if (TP + FN != 0):
            eval_dict[conf_threshold]['powerdrill']['recall'] = len(powerdrill_TP_df) / (len(powerdrill_TP_df) + len(powerdrill_FN_df))

        eval_dict[conf_threshold]['screwdriver'] = {}
        eval_dict[conf_threshold]['screwdriver']['TP'] = len(screwdriver_TP_df)
        eval_dict[conf_threshold]['screwdriver']['FP'] = len(screwdriver_FP_df)
        eval_dict[conf_threshold]['screwdriver']['FN'] = len(screwdriver_FN_df)
        SCR_TP = len(screwdriver_TP_df)
        SCR_FP = len(screwdriver_FP_df)
        SCR_FN = len(screwdriver_FN_df)
        screw_undefined = True
        if ((SCR_TP + SCR_FP != 0) & (SCR_TP + SCR_FN != 0)):
            screw_undefined = False
            eval_dict[conf_threshold]['screwdriver']['precision'] = len(screwdriver_TP_df) / (len(screwdriver_TP_df) + len(screwdriver_FP_df))
        #if (TP + FN != 0):
            eval_dict[conf_threshold]['screwdriver']['recall'] = len(screwdriver_TP_df) / (len(screwdriver_TP_df) + len(screwdriver_FN_df))
        
        if (power_undefined or screw_undefined):
            output += (f"Confidence-Threshold: {conf_threshold:.2f} | Recall: Undefined | Precision: Undefined "
            f"| TP: {len(powerdrill_TP_df) + len(screwdriver_TP_df):4} | FN: {len(powerdrill_FN_df) + len(screwdriver_FN_df):4} | FP: {len(powerdrill_FP_df) + len(screwdriver_FP_df):4} \n")
            continue    

        eval_dict[conf_threshold]['total'] = {}
        eval_dict[conf_threshold]['total']['precision'] = (eval_dict[conf_threshold]['screwdriver']['precision'] + eval_dict[conf_threshold]['powerdrill']['precision']) / 2
        eval_dict[conf_threshold]['total']['recall'] = (eval_dict[conf_threshold]['screwdriver']['recall'] + eval_dict[conf_threshold]['powerdrill']['recall']) / 2

        output += (f"Confidence-Threshold: {conf_threshold:.2f} | Recall: {round(eval_dict[conf_threshold]['total']['recall'], 2):.2f} | Precision: {round(eval_dict[conf_threshold]['total']['precision'], 2):.2f} "
            f"| TP: {len(powerdrill_TP_df) + len(screwdriver_TP_df):4} | FN: {len(powerdrill_FN_df) + len(screwdriver_FN_df):4} | FP: {len(powerdrill_FP_df) + len(screwdriver_FP_df):4} \n")
        

    #prec-rec for total
    total_precision_list = []
    total_recall_list = []
    pow_recall_list = []
    pow_precision_list = []
    screw_recall_list = []
    screw_precision_list = []
    for conf_threshold in eval_dict:
        if 'total' in eval_dict[conf_threshold]:
            total_recall_list.append(eval_dict[conf_threshold]['total']['recall'])
            total_precision_list.append(eval_dict[conf_threshold]['total']['precision'])
        if (('recall' in eval_dict[conf_threshold]['powerdrill']) and ('precision' in eval_dict[conf_threshold]['powerdrill'])):
            pow_recall_list.append(eval_dict[conf_threshold]['powerdrill']['recall'])
            pow_precision_list.append(eval_dict[conf_threshold]['powerdrill']['precision'])
        if (('recall' in eval_dict[conf_threshold]['screwdriver']) and ('precision' in eval_dict[conf_threshold]['screwdriver'])):
            screw_recall_list.append(eval_dict[conf_threshold]['screwdriver']['recall'])
            screw_precision_list.append(eval_dict[conf_threshold]['screwdriver']['precision'])

    # same method as above
    #np_pow_recall_list = np.array(pow_recall_list)
    #np_pow_prec_list = np.array(pow_precision_list)
    #sorted_powerdrill = np.argsort(np_pow_recall_list)
    #sorted_pow_recall = np_pow_recall_list[sorted_powerdrill]
    #sorted_pow_prec = np_pow_prec_list[sorted_powerdrill]
    #AP_powerdrill = metrics.auc(sorted_pow_recall, sorted_pow_prec)

    #np_screw_recall_list = np.array(screw_recall_list)
    #np_screw_prec_list = np.array(screw_precision_list)
    #sorted_screwdriver = np.argsort(np_screw_recall_list)
    #sorted_screw_recall = np_screw_recall_list[sorted_screwdriver]
    #sorted_screw_precision = np_screw_prec_list[sorted_screwdriver]
    #AP_screwdriver = metrics.auc(sorted_screw_recall, sorted_screw_precision)

    #mAP = 0.5 * (AP_powerdrill + AP_screwdriver)
    #output += (f"mAP: {mAP} \n\n\n-------------------------------\n\n\n")

    interpolated_pow_prec = interpolate_prec(pow_precision_list)
    interpolated_screw_prec = interpolate_prec(screw_precision_list)
    interpolated_total_prec = interpolate_prec(total_precision_list)

    if PLOT:
        fig, ax = plt.subplots()
        ax.plot(total_recall_list, interpolated_total_prec,marker='o', linestyle='-')
        ax.set_title("Total Precision-Recall Curve : YOLO-" + str(p.parent.name))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True)
        fig.savefig(str(p.parent) + '/' + str(p.parent.name) + '_total-prec-rec.png', dpi=300)
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(pow_recall_list, interpolated_pow_prec, marker='o', linestyle='-')
        ax.set_title("Powerdrill Precision-Recall Curve : YOLO-" + str(p.parent.name))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True)
        fig.savefig(str(p.parent) + '/' + str(p.parent.name) + '_power-prec-rec.png', dpi=300)
        plt.close()

        fig,ax = plt.subplots()
        ax.plot(screw_recall_list, interpolated_screw_prec, marker='o', linestyle='-')
        ax.set_title("Screwdriver Precision-Recall Curve : YOLO-" + str(p.parent.name))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True)
        fig.savefig(str(p.parent) + '/' + str(p.parent.name) + '_screw-prec-rec.png', dpi=300)
        plt.close()
    
    if args.simple_output:
        output = ''
    ap = 0.0
    for precision in interpolated_total_prec:
        ap += precision
    ap = ap / 101
    output += f"AP@[{iou_threshold:.2f}]={round(ap,2)}\n"
    return output, ap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--yolo", required=False ,default=False, type=bool)
    parser.add_argument("--yolact", required=False,default=False, type=bool)
    parser.add_argument("--cluster", required=False, default=False, type=bool)
    parser.add_argument("--plot", required=False, default=False, type=bool)
    parser.add_argument("--simple_output", required=False, default=False, type=bool)
    args = parser.parse_args()
    if args.yolo:
        if args.cluster:
            parent_path = ''
        else:
            parent_path = '/Users/kerim/dev/BachelorThesis/results_eval/YOLO'
        yolo_folders = os.listdir(parent_path)
        for folder in yolo_folders:
            if (folder == '.DS_Store'):
                continue
            eval_file_path = parent_path + '/' + folder + '/eval_values.csv'
            threshold_list = list(range(50,100,5))
            threshold_list = [x/100 for x in threshold_list]
            output = f"Model {folder}:\n-------\n"
            ap_all = 0
            for iou_threshold in threshold_list:
                temp_output, temp_ap = eval_yolo(eval_file_path, iou_threshold, PLOT=args.plot)
                output += temp_output
                ap_all += temp_ap
            ap_all = ap_all / 10
            output += f"AP@[0.5:0.95]={round(ap_all,2)}\n"
            with open(parent_path + '/' + folder + '/eval_output.txt', 'w') as f:
                f.write(output)
            print(output)
    if args.yolact:
        if args.cluster:
            parent_path = ''
        else:
            parent_path = '/Users/kerim/dev/BachelorThesis/results_eval/YOLACT'
        yolact_folders = os.listdir(parent_path)
        for folder in yolact_folders:
            if (folder == '.DS_Store'):
                continue
            eval_file_path = parent_path + '/' + folder + '/eval_values.csv'
            threshold_list = list(range(50,100,5))
            threshold_list = [x/100 for x in threshold_list]
            output = f"Model {folder}:\n-------\n"
            ap_all = 0
            for iou_threshold in threshold_list:
                temp_output,temp_ap = eval_yolact(eval_file_path, iou_threshold)
                output += temp_output
                ap_all += temp_ap
            ap_all = ap_all / 10
            output += f"AP@[0.5:0.95]={round(ap_all,2)}\n"
            with open(parent_path + '/' + folder + '/eval_output.txt', 'w') as f:
                f.write(output)
            print(output)
    print("OK!")