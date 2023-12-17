#!/bin/bash
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python/3.8.5 gdal/3.1.2 geos/3.6.2
#source /cluster/home/kbirgi/myenv/bin/activate
python3 code/MakeCoco.py --parent_path=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp/test_orx --folders=[\"000000\",\"000002\",\"000004\",\"001001\",\"001003\",\"000001\",\"000003\",\"001000\",\"001002\",\"001004\"] --output_file=testORX
#deactivate
