import os
import json 

def createPredsDict(bbox_preds_path, mask_preds_path, test_annotations_path):
    final_dict = {}
    f = open(test_annotations_path)
    annotations_dict = json.load(f)
    f.close()

    f = open(bbox_preds_path)
    bbox_pred_dict_pre = json.load(f)
    f.close()
    bbox_pred_dict = {}
    for pred_dict in bbox_pred_dict_pre:
        img_id = pred_dict['image_id']
        bbox_pred_dict[img_id] = []
        temp_dict = {}
        temp_dict['category_id'] = pred_dict['category_id']
        temp_dict['bbox'] = pred_dict['bbox']
        temp_dict['score'] = pred_dict['score']
    

    for image in annotations_dict['images']:
        image_path = image['file_name']
        image_path = '/Users/kerim/Desktop/' + image_path
        image['file_name'] = image_path 
        image['bbox_pred'] = {}
        
        
        print(image)
        exit()

    
if __name__ == "__main__":
    print('BEGINNING PROGRAM!')
    bbox_preds_path = '/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/results/bbox_detections.json'
    mask_preds_path = '/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/results/mask_detections.json'
    test_annotations_path = '/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/test_annotations.json'
    output_file = '/Users/kerim/dev/BachelorThesis/Annotations/testSetSubsetSSD/'
    createPredsDict(bbox_preds_path, mask_preds_path, test_annotations_path)
    print('OK!')