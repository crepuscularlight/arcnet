# Custom training for Detectron2 model using the TACO dataset.

# Importing general libraries
import json
import random
import cv2
import os
import argparse

#Importing custom functions
from utils import *

#Importing Detectron libraries
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# To train the model:
#    python taco_train.py train

# To test the model:
#    python taco_train.py test


# Parsing global arguments
parser = argparse.ArgumentParser(description='Custom implementation of Detectron2 using the TACO dataset.')
parser.add_argument('--class_map', required=False, metavar="/path/file.csv", help='Target classes')
parser.add_argument('--image_path', required=False, metavar="/path/file.jpg", help='Test image')
parser.add_argument("command", metavar="<command>",help="Opt: 'train', 'evaluate', 'test'")
args = parser.parse_args()

# TODO Re-map dataset classes to map.csv file
class_map = {}
map_to_one_class = {}


# TODO Create new train/test/val split for training every time this script is called.


# Registering the custom dataset using Detectron2 libraries
register_coco_instances("taco_train",{},"./data/annotations_0_train.json","./data")
register_coco_instances("taco_test",{},"./data/annotations_0_test.json","./data")
register_coco_instances("taco_val",{},"./data/annotations_0_val.json","./data")

dataset_dicts_train = DatasetCatalog.get("taco_train")
dataset_dicts_test = DatasetCatalog.get("taco_test")
dataset_dicts_val = DatasetCatalog.get("taco_val")

taco_metadata = MetadataCatalog.get("taco_train")
print("registered successfully")

# verify the custom dataset was imported successfully.
for d in random.sample(dataset_dicts_train, 2):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=taco_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)

    # image too large to display - resize down to fit in the screen
    img_new = out.get_image()[:, :, ::-1]
    img_resized = ResizeWithAspectRatio(img_new, width=800)
    cv2.imshow("rand_name", img_resized)# out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Training custom dataset
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("taco_train",)
cfg.DATASETS.TEST = ("taco_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 264   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 60  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
if args.command == "train":
    trainer.train()

elif args.command == 'test':
    # Inference should use the config with parameters that are used in training
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    for d in random.sample(dataset_dicts_val, 10):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=taco_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2.imshow(out.get_image()[:, :, ::-1])

        # image too large to display - resize down to fit in the screen
        img_out = out.get_image()[:, :, ::-1]
        img_out_resized = ResizeWithAspectRatio(img_out, width=800)
        cv2.imshow("rand_name", img_out_resized)  # out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Further testing and validation.
        evaluator = COCOEvaluator("taco_val", ("bbox", "segm"), False, output_dir="./output/")
        val_loader = build_detection_test_loader(cfg, "taco_val")
        print(inference_on_dataset(trainer.model, val_loader, evaluator))
        # another equivalent way to evaluate the model is to use `trainer.test`


elif args.image_path == 'custom':
    im = cv2.imread(args.image_path)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=taco_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('./output/', out)
    # cv2.imshow(out.get_image()[:, :, ::-1])



