"""
Author: Fidel Esquivel Estay
GH: phideltaee
Description: Custom training model for Detectron2 using a modified version of the TACO dataset.

------------------------------------------------------
------------------------------------------------------
NOTES on Implementation:

    # Training on TACO dataset. First step is to remap the directory to the desired number of classes. Choose a map
    (dictionary) or create your own and place it in the folder 'maps'
    # Run the remapping
    #   python remap_classes --class_map <path_to_map/file.csv> --ann_dir <path_to_annotations/file.json>

    # To train the model:
    #    python arcnet_main.py train --dataset_dir <path_to_dataset/>

    # To test the model:
    #    python arcnet_main.py test

    # To try the model for predicting an image
    #    python arcnet_main.py inference --image_path path_to_test_image.jpg


    # First make sure you have split the dataset into train/val/test set. e.g. You should have annotations_0_train.json
    # in your dataset dir.
    # Otherwise, You can do this by calling
    python3 split_dataset.py --dataset_dir ../data

    # Train a new model starting from pre-trained COCO weights on train set split #0
    python3 -W ignore detector.py train --model=coco --dataset=../data --class_map=./taco_config/map_10.csv --round 0

    # Continue training a model that you had trained earlier
    python3 -W ignore detector.py train  --dataset=../data --model=<model_name> --class_map=./taco_config/map_10.csv --round 0

    # Continue training the last model you trained with image augmentation
    python3 detector.py train --dataset=../data --model=last --round 0 --class_map=./taco_config/map_10.csv --use_aug

    # Test model and visualize predictions image by image
    python3 detector.py test --dataset=../data --model=<model_name> --round 0 --class_map=./taco_config/map_10.csv

    # Run COCO evaluation on a trained model
    python3 detector.py evaluate --dataset=../data --model=<model_name> --round 0 --class_map=./taco_config/map_10.csv



    # Check Tensorboard for model training validation information.
     tensorboard --logdir ./output/

"""
# Importing general libraries
import json
import random
import cv2
import os
import argparse
import time
from datetime import datetime

# Importing custom functions
from utils import *

# Importing Detectron libraries
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

# Parsing global arguments
parser = argparse.ArgumentParser(description='Custom implementation of Detectron2 using the TACO dataset.')
parser.add_argument('--class_map', required=False, metavar="/path/file.csv", help='Target classes')
parser.add_argument('--image_path', required=False, metavar="/path/file.jpg", help='Test image')
parser.add_argument('--data_dir', required=False, metavar="/path_to_data/", help='Dataset directory')
parser.add_argument("command", metavar="<command>",help="Opt: 'train', 'test', 'inference")
args = parser.parse_args()


# TODO Create new train/test/val split for training every time this script is called.

# Registering the custom dataset using Detectron2 libraries
# TODO add dataset directory and map version to data loading part.
#register_coco_instances("taco_train",{},"./data/annotations_0map2_train.json","./data")
register_coco_instances("taco_train",{},"./data/annotations_0_train.json","./data")
register_coco_instances("taco_test",{},"./data/annotations_0_test.json","./data")
register_coco_instances("taco_val",{},"./data/annotations_0_val.json","./data")

dataset_dicts_train = DatasetCatalog.get("taco_train")
dataset_dicts_test = DatasetCatalog.get("taco_test")
dataset_dicts_val = DatasetCatalog.get("taco_val")

taco_metadata = MetadataCatalog.get("taco_train")
print("datasets registered successfully")

# verify the custom dataset was imported successfully by loading some images
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
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 264   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 60  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
if args.command == "train":
    # Train the trainer with the configuration set earlier.
    trainer.train()

elif args.command == 'test':
    # Inference should use the config with parameters that are used in training
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
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


elif args.command == 'inference':
    print(args.image_path)

    # Inference should use the config with parameters that are used in training
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    im = cv2.imread(args.image_path)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=taco_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img_out = out.get_image()#[:, :, ::-1]


    time_out = get_timestamp()

    cv2.imwrite('./prediction'+time_out+'.jpg', img_out)
    # to visualize output uncomment below
    # cv2.imshow(out.get_image()[:, :, ::-1])
