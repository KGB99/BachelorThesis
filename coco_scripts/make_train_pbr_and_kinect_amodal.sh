#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=24:00:00
module load gcc/8.2.0 python/3.8.5 gdal/3.1.2 geos/3.6.2
python3 code/makeCoco.py --parent_path=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp --folders="[\"train_pbr_random_lighting/000000\", \"train_pbr_random_lighting/000001\", \"train_pbr_random_lighting/000002\", \"train_pbr_random_lighting/000003\", \"train_pbr_random_lighting/000004\", \"train_pbr_random_lighting/000005\", \"train_pbr_random_lighting/000006\", \"train_pbr_random_lighting/000007\", \"test_pbr_random_lighting/000000\", \"train_kinect_pbr/000000\", \"train_kinect_pbr/000001\", \"train_kinect_pbr/000002\", \"train_kinect_pbr/000003\", \"train_kinect_pbr/000004\", \"train_kinect_pbr/000005\", \"train_kinect_pbr/000006\", \"train_kinect_pbr/000007\", \"train_kinect_pbr/000008\", \"train_kinect_pbr/000009\", \"train_kinect_pbr/000010\", \"train_kinect_pbr/000011\", \"train_kinect_pbr/000012\"]" --output_dir=train_pbr_random_and_kinect --output_file=amodal_labels --img_file_type=jpg --bitmask_file_type=png --amodal=True --path_splitter=mvpsp
