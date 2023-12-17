#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=40G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.8.5 gdal/3.1.2 geos/3.6.2
python3 code/MakeCoco.py --parent_path=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp/test --folders="[\"004000\",\"004004\",\"004006\",\"005003\",\"005005\",\"009002\",\"009004\",\"020001\",\"020005\"]" --output_file=testSubset --img_file_type=png --bitmask_file_type=png
