import fiftyone as fo

name = "001000-train"
#data_path = "/Volumes/cvg-ssd-05/mvpsp/train/001000/rgb"
data_path = "/Users/kerim/dev/BachelorThesis/Data_subset/mvpsp/train/rgb"
label_path = "/Users/kerim/dev/BachelorThesis/Annotations/subset_test.json"

#print datasets
#print(fo.list_datasets())

#import dataset
dataset = fo.Dataset.from_dir(
    dataset_type = fo.types.COCODetectionDataset,
    data_path = data_path,
    labels_path = label_path,
    name = name,
    )

session = fo.launch_app(dataset, port=5151)
session.wait()
