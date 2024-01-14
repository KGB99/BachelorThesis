#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.8.5 gdal/3.1.2 geos/3.6.2
#source /cluster/home/kbirgi/myenv/bin/activate
python3 code/makeCoco.py --multiple_folders=True --parent_path=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp --folders="[\"train_hololens_pbr\", \"train_kinect_pbr\"]" --output_dir=trainpbr --output_file=amodal_labels --img_file_type=jpg --bitmask_file_type=png --amodal=True
#deactivate
