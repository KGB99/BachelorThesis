#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.10.4

python3 code/evaluateIOU.py --images_dir=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp --preds_path=/cluster/home/kbirgi/results/orx_subset_results --labels_path=/cluster/home/kbirgi/Annotations/subsets/testSubSubsetORX_results/test_annotations.json --output=testSubSubsetORX_results
