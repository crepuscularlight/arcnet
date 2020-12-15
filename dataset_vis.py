import cv2
import random
import os
from detectron2.utils.visualizer import Visualizer

# Importing Detectron2 libraries
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# Importing the new dataset json file and specifying data content directory
dataset_path = "test_dataset/arc_litter-v1.1_coco.json"
data_path = "test_dataset/segments/festay_arc_litter/v1.1"

# # importing the dataset in COCO format
#import json
#with open(dataset_path) as f:
#    dataset = json.load(f)

# Registering the dataset using the detectron2 format to validate correctness of labels
register_coco_instances("taco_train",{},dataset_path,data_path)
dataset_dicts_train = DatasetCatalog.get("taco_train")
taco_metadata = MetadataCatalog.get("taco_train")
print("datasets registered successfully")

# verify the custom dataset was imported successfully by loading some images
for d in random.sample(list(dataset_dicts_train), 1):
    print(d["file_name"])
    assert os.path.isfile(d["file_name"]), "Image not loaded correctly!"
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=taco_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)

    # image too large to display - resize down to fit in the screen
    img_new = out.get_image()[:, :, ::-1]
    img_resized = img_new#ResizeWithAspectRatio(img_new, width=800)
    cv2.imshow("rand_name", img_resized)# out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

