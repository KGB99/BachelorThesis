#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.10.4

python3 code/evaluate.py --images_dir=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp --preds_path=/cluster/project/infk/cvg/heinj/students/kbirgi/BachelorThesis/Annotations/quantitative_evaluation/preds --bbox_preds=pbr_random_and_kinect_bbox_detections.json --mask_preds=pbr_random_and_kinect_mask_detections.json --labels_path=/cluster/project/infk/cvg/heinj/students/kbirgi/BachelorThesis/Annotations/quantitative_evaluation/test_labels/test_annotations_150.json --output=quantitative_evaluation --save_images=True
