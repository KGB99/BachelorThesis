#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.8.5
python3 code/makeAnnotations.py --output_dir=quant_eval_trials_nogt/test_labels --coco_file=/cluster/project/infk/cvg/heinj/students/kbirgi/BachelorThesis/Annotations/quant_eval_trials_nogt/amodal_labels_150.json --test=True
