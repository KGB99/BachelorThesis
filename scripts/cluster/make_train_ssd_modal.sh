#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.8.5 gdal/3.1.2 geos/3.6.2
#source /cluster/home/kbirgi/myenv/bin/activate
python3 code/makeCoco.py --parent_path=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp/train --folders="[\"001000\", \"001001\", \"001002\", \"001003\", \"001004\", \"001005\", \"001006\", \"016000\", \"016001\", \"016002\", \"016003\", \"016004\", \"016005\", \"016006\"]" --output_dir=trainSSD --output_file=all_labels --img_file_type=png --bitmask_file_type=png --amodal=False
#deactivate