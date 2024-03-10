#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.10.4

python3 code/preprocess_eval.py --yolact=True --images_dir=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp --preds_path=/cluster/project/infk/cvg/heinj/students/kbirgi/BachelorThesis/Annotations/quant_eval_trials_nogt/preds/yolact/pbr_new_30000 --bbox_preds=pbr_30000_bbox_detections.json --mask_preds=pbr_30000_mask_detections.json --labels_path=/cluster/project/infk/cvg/heinj/students/kbirgi/BachelorThesis/Annotations/quant_eval_trials_nogt/amodal_labels_150.json --output=yolact/pbr_new_30000 --save_images=False
