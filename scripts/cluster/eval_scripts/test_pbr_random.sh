#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.10.4

python3 code/evaluate.py --images_dir=/cluster/project/infk/cvg/heinj/datasets/bop/pbr_random_lighting --preds_path=/cluster/project/infk/cvg/heinj/students/kbirgi/BachelorThesis/Annotations/test_pbr_random/eval/preds --labels_path=/cluster/project/infk/cvg/heinj/students/kbirgi/BachelorThesis/Annotations/test_pbr_random/test_labels/amodal/test_annotations.json --output=test_pbr_random
