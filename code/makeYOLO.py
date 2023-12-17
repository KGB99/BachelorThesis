import argparse
import os
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--train_dir', required=True, type=str)
    parser.add_argument('--labels_dir', required=True, type=str)
    parser.add_argument("--coco_file", help="the path to the coco file containing all infos", required=True, type=str)
    args = parser.parse_args()

    f = open(args.coco_file, 'r')
    coco_dict = json.load(f)
    f.close()

    print(coco_dict)
