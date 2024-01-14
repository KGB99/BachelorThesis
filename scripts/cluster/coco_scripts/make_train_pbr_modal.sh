#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.8.5 gdal/3.1.2 geos/3.6.2
#source /cluster/home/kbirgi/myenv/bin/activate
python3 code/makeCoco.py --parent_path=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp/train_pbr --folders="[\"000000\", \"000001\", \"000002\", \"000003\", \"000004\", \"000005\", \"000006\", \"000007\", \"000008\", \"000009\", \"000010\", \"000011\", \"000012\", \"001000\", \"001001\", \"001002\", \"001003\", \"001004\", \"001005\", \"001006\", \"001007\", \"001008\", \"001009\", \"001010\", \"001011\", \"001012\"]" --output_dir=trainpbr --output_file=modal_labels --img_file_type=jpg --bitmask_file_type=png --amodal=False
#deactivate