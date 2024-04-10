#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.8.5
python3 code/makeAnnotations.py --output_dir=supervised_pbr_base --coco_file=/cluster/project/infk/cvg/heinj/students/kbirgi/Annotations/gen_annotations/stride_50_pbr_base_30000/generated_labels.json --old_annotations=True
