#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=24:00:00
module load gcc/8.2.0 python/3.8.5 gdal/3.1.2 geos/3.6.2
python3 code/makeCoco.py --parent_path=/cluster/project/infk/cvg/heinj/datasets/bop/pbr_random_lighting/test --folders="[\"000001\"]" --output_dir=test_Pbr_random --output_file=amodal_labels --img_file_type=jpg --bitmask_file_type=png --amodal=True --path_splitter=pbr_random_lighting
