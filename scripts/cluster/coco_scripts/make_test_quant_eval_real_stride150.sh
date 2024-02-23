#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=24:00:00
module load gcc/8.2.0 python/3.8.5 gdal/3.1.2 geos/3.6.2
python3 code/makeCoco.py --parent_path=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp --folders="[\"test/004000\", \"test/004001\", \"test/004002\", \"test/004003\", \"test/004004\", \"test/004005\",  \"test/004006\", \"test/005000\", \"test/005001\", \"test/005002\", \"test/005003\", \"test/005004\", \"test/005005\", \"test/005006\", \"test/009000\", \"test/009001\", \"test/009002\", \"test/009003\", \"test/009004\", \"test/009005\", \"test/009006\", \"test/020000\", \"test/020001\", \"test/020002\", \"test/020003\", \"test/020004\", \"test/020005\", \"test/020006\"]" --output_dir=quantitative_evaluation --output_file=amodal_labels_150 --img_file_type=png --bitmask_file_type=png --amodal=True --path_splitter=mvpsp --stride=150
