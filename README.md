# Labels
## Creating COCO labels
To create coco labels run the preferred scripts from the coco_scripts folder. The names for the sets are more or less self explanatory. These scripts simply store all relevant values in a json but not in the final form that can be used for model training. This is done to save computation cost. This way the relevant values are calculated once and the formatting can be adapted to whatever is necessary. So after running these scripts there are scripts seperately for yolact and yolo. The label_scripts folder contains scripts for yolact and the yolo_scripts folder contains scripts for yolo since both need the data in a unique format.

# Evaluating predictions
For evaluation there are two steps. The first is the preprocessing and the second is the actual evaluation.
The scripts inside of eval_scripts can be used to run the preprocessing to calculate IoU values of the masks from GT and predictions as well as DSC or PA. Be sure to store the predictions that come out of the yolact and yolo models where specified by the scripts or to adapt the directory that is noted in the scripts.

Finally the relevant metrics such as AP can be calculated after the preprocessing.
Running yolact_eval.sh will evaluate all the values saved in the results_eval/yolact folder and running yolo_eval.sh will evaluate all the values in the results_eval/yolo folder