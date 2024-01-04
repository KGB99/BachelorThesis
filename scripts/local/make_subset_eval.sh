#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.8.5 geos/3.6.2
#source /cluster/home/kbirgi/myenv/bin/activate
python3 code/MakeCoco.py --parent_path=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp/test_orx --folders=[\"000000\"] --output_file=testSubSubsetORX --img_file_type=jpg --bitmask_file_type=png
#deactivate
