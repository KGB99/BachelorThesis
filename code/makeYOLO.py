import argparse
import os
import json
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--train_dir', required=True, type=str)
    parser.add_argument('--labels_dir', required=True, type=str)
    parser.add_argument('--img_type', required=True, type=str)
    parser.add_argument("--coco_file", help="the path to the coco file containing all infos", required=True, type=str)
    parser.add_argument("--create_test", required=False, type=bool, default=False)
    parser.add_argument("--info_path", required=True, type=str)
    parser.add_argument("--output", required=False, default="output", type=str)
    args = parser.parse_args()
    train_dir = args.train_dir
    labels_dir = args.labels_dir
    coco_file = args.coco_file
    img_type = args.img_type
    create_test = args.create_test
    info_path = args.info_path
    output_file = args.output

    train_split = 0.7
    
    if (not os.path.exists(labels_dir)):
        os.mkdir(labels_dir)

    f = open(coco_file, 'r')
    coco_dict = json.load(f)
    f.close()

    for i,camera in enumerate(coco_dict):
        print("Camera: " + str(i) + "/" + str(len(coco_dict)))
        camera_dict = coco_dict[camera]
        split_counter = 1
        for j,image in enumerate(camera_dict):
            print("Camera: " + str(i) + "/" + str(len(coco_dict)) + " | Image: " + str(j) + "/" + str(len(camera_dict)))
            camera_len = len(camera_dict.keys())
            train_limit = int(train_split * camera_len)
            img_dict = camera_dict[image]['img']
            mask_dict = camera_dict[image]['mask']
            category_id = mask_dict['category_id']
            if (not category_id in [1,2]):
                continue

            img_path = train_dir + "/" + img_dict['file_name']
            
            img_width = img_dict['width']
            img_height = img_dict['height']

            bbox_x = int(mask_dict['bbox'][0])
            bbox_y = int(mask_dict['bbox'][1])
            bbox_w = int(mask_dict['bbox'][2])
            bbox_h = int(mask_dict['bbox'][3])

            bbox_center_x_norm = (bbox_x + (bbox_w / 2)) / img_width
            bbox_center_y_norm = (bbox_y + (bbox_h / 2)) / img_height
            bbox_w_norm = bbox_w / img_width
            bbox_h_norm = bbox_h / img_height

            #print(bbox_center_x_norm, bbox_center_y_norm, bbox_w_norm, bbox_h_norm)
            #image_processed = cv2.imread(img_path)
            #image_processed = cv2.circle(image_processed, center=(bbox_center_x, bbox_center_y), radius=10, color=(0, 0, 255), thickness=-1)
            #cv2.imshow("image_preview", image_processed)
            #cv2.waitKey(0)
            
            temp_img_path = img_dict['file_name']
            temp_array = []
            while (os.path.split(temp_img_path)[1] != ''):
                #print(temp_img_path)
                temp_path_split = os.path.split(temp_img_path)
                temp_array.append(temp_path_split[1])
                temp_img_path = temp_path_split[0]
            temp_array.reverse()
            temp_path = labels_dir
            for i in range(len(temp_array) - 1):
                temp_path = temp_path + "/" + temp_array[i]
                if (not os.path.exists(temp_path)):
                    os.mkdir(temp_path)

            label_path = str.replace((labels_dir + "/" + img_dict['file_name']), ("." + img_type), ".txt")
            f = open(label_path, "a")
            # YOLO starts category indexing at 0 so we need to do minus 1
            f.write(str(category_id - 1) + " " + str(bbox_center_x_norm) + " " + str(bbox_center_y_norm) + " " + str(bbox_w_norm) + " " + str(bbox_h_norm) + "\n")
            f.close()

            if not create_test:
                if (split_counter < train_limit):
                    f = open(info_path + "/" + output_file + "_train.txt", "a")
                    f.write(img_path + "\n")
                    f.close()
                    split_counter += 1
                else:
                    f = open(info_path + "/" + output_file + "_val.txt", "a")
                    f.write(img_path + "\n")
                    f.close()
            else:
                f = open(info_path + "/" + output_file + "_test.txt", "a")
                f.write(img_path + "\n")
                f.close()
            
    print("OK!")