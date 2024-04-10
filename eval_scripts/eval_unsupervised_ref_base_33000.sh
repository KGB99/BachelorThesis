#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.10.4

python3 code/preprocess_eval.py --yolact=True --images_dir=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp --preds_path=/cluster/project/infk/cvg/heinj/students/kbirgi/BachelorThesis/Annotations/quant_eval_trials_nogt/preds/yolact/ref_unsupervised_base_33000 --bbox_preds=ref_unsupervised_base_33000_bbox_detections.json --mask_preds=ref_unsupervised_base_33000_mask_detections.json --labels_path=/cluster/project/infk/cvg/heinj/students/kbirgi/BachelorThesis/Annotations/quant_eval_trials_nogt/amodal_labels_150.json --output=yolact/ref_unsupervised_base_33000 --save_images=False
