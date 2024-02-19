#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.8.5
python3 code/makeAnnotations.py --output_dir=refinement_real/5000_labels/train_labels --coco_file=/cluster/project/infk/cvg/heinj/students/kbirgi/Annotations/refinement_real/5000_labels/016006_5000_amodal_labels.json --train_ratio=1.0 --with_val=False
