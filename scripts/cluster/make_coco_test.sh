#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=24:00:00
module load gcc/8.2.0 python/3.8.5 gdal/3.1.2 geos/3.6.2
python3 code/MakeCoco.py --parent_path=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp/test --folders="[\"004000\",\"004001\",\"004002\",\"004003\",\"004004\",\"004005\",\"004006\",\"005000\",\"005001\",\"005002\",\"005003\",\"005004\",\"005005\",\"005006\",\"009000\",\"009001\",\"009002\",\"009003\",\"009004\",\"009005\",\"009006\",\"020000\",\"020001\",\"020002\",\"020003\",\"020004\",\"020005\",\"020006\"]" --output_file=testAll --img_file_type=png --bitmask_file_type=png
