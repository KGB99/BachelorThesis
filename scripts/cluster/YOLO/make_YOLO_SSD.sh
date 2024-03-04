#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=100G
#SBATCH --time=4:00:00

module load gcc/8.2.0 python/3.8.5
python3 code/makeYOLO.py --train_dir=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp --labels_dir=/cluster/home/kbirgi/Annotations/yoloSSD/mvpsp --coco_file=/cluster/project/infk/cvg/heinj/students/kbirgi/BachelorThesis/Annotations/trainSSD/modal_labels.json --img_type=png --info_path=/cluster/home/kbirgi/Annotations/yoloSSD --val_folders="[\"test_pbr_random_lighting/000000\"]"
