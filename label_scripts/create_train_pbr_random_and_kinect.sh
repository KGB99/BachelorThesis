#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.8.5
python3 code/makeAnnotations.py --output_dir=train_pbr_random_and_kinect/train_labels/amodal --coco_file=/cluster/project/infk/cvg/heinj/students/kbirgi/BachelorThesis/Annotations/train_pbr_random_and_kinect/amodal_labels.json --val_folders="[\"test_pbr_random_lighting/000000\"]"
