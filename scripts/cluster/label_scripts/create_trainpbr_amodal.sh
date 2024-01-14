#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.8.5
python3 code/makeAnnotations.py --output_dir=trainpbr/train_labels/amodal --coco_file=/cluster/project/infk/cvg/heinj/students/kbirgi/BachelorThesis/Annotations/trainpbr/amodal_labels.json
