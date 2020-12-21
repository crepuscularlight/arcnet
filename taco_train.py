"""
Author: Fidel Esquivel Estay
GitHub: phideltaee
Description: Custom training model for Detectron2 using a modified version of the TACO dataset.

------------------------------------------------------
------------------------------------------------------
NOTES on Implementation:

    # Training on TACO dataset.

    # Step 1:   Remap output to the desired number of classes. Choose a map (dictionary) or create
    #           your own and place it in the folder 'maps'.

    #   - - Run the remapping - -
    #           python remap_classes --class_map <path_to_map/file.csv> --ann_dir <path_to_annotations/file.json>

    # Step 2:   Split dataset into train-test splits, k-times for k-fold cross validation.

    #   - - Split the dataset - -
    #           python split_dataset.py --nr_trials <K_folds> --out_name <name of file> --dataset_dir <path_to_data>

    # To train the model:
    #    Template: python arcnet_main.py --class_map <mapto_class.csv> --data_dir <path_to_dataset/> train
    #    EXAMPLE:  python arcnet_main.py --class_map maps/map_to_2.csv --data_dir data train

    # To test the model:
    #    Template:  python arcnet_main.py test --weights <path_to/weigts.pth>
    #    EXAMPLE:   python arcnet_main.py test --weights output/taco_3000.pth

    # To try the model for inference an image
    #    TEMPLATE: python arcnet_main.py inference --image_path <path/to/test_image.jpg> --weights <path_to/weights.pth>
    #    EXAMPLE:  python arcnet_main.py inference --image_path img_test/trash_01.jpg --weights output/taco_3000.pth

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

# Importing Detectron2 libraries
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
parser.add_argument('--image_path', required=False, default='./img_test/test_img1.jpg',  metavar="/path/file.jpg", help='Test image')
parser.add_argument('--data_dir', required=False, default='./data', metavar="/path_to_data/", help='Dataset directory')
parser.add_argument("command", metavar="<command>",help="Opt: 'train', 'test', 'inference")
parser.add_argument('--weights', required=False, default='./output/model_final.pth', metavar="/trained_weights.pth", help='weights')
args = parser.parse_args()

# TODO Create new train/test/val split for training every time this script is called.

# Registering the custom dataset using Detectron2 libraries

# TODO load train test and val data directly from the run commands (not hardcoded)
# gets the annotation directly from the train set.
#class_map_name = args.class_map.split("/")[-1].split(".")[-2]

# Registering with 2 classes - map_to_2 annotations
register_coco_instances("taco_train",{},"./data/annotations_0_map_2_train.json","./data")
register_coco_instances("taco_test",{},"./data/annotations_0_map_2_test.json","./data")
register_coco_instances("taco_val",{},"./data/annotations_0_map_2_val.json","./data")

# Registering with standard TACO annotations - 60 classes.
#register_coco_instances("taco_train",{},"./data/annotations_0__train.json","./data")
#register_coco_instances("taco_test",{},"./data/annotations_0__test.json","./data")
#register_coco_instances("taco_val",{},"./data/annotations_0__val.json","./data")

# Obtaining the dataset catalog for each, train, val and test.
dataset_dicts_train = DatasetCatalog.get("taco_train")
#dataset_dicts_test = DatasetCatalog.get("taco_test")
dataset_dicts_val = DatasetCatalog.get("taco_val")

# Adding custom test file for ARC Litter dataset
register_coco_instances("arc_test",{},"./segments/festay_arc_litter/arc_litter-v1.1_coco.json", "./segments/festay_arc_litter/v1.1")
dataset_dicts_test = DatasetCatalog.get("arc_test")

taco_metadata = MetadataCatalog.get("taco_train")
print("datasets registered successfully")

# verify the custom dataset was imported successfully by loading some images
for d in random.sample(dataset_dicts_train, 1):
    print(d["file_name"])
    assert os.path.isfile(d["file_name"]), "Image not loaded correctly!"
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
# Validation Set (ignoring wrong naming convention)
cfg.DATASETS.TEST = ("arc_test",)
cfg.TEST.EVAL_PERIOD = 60

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.0025  # Starting lr. Adaptive.
cfg.SOLVER.MAX_ITER = 600
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Freeze the first several stages so they are not trained.
# There are 5 stages in ResNet. The first is a convolution, and the following
# stages are each group of residual blocks.
cfg.MODEL.BACKBONE.FREEZE_AT = 4 #default is 4

# default trainer, does not include validation loss. Custom coco trainer created to tackle this.
#trainer = DefaultTrainer(cfg)

# Training with custom validation loss trainer CocoTrainer.py
from CocoTrainer import CocoTrainer
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)

if args.command == "train":
    trainer.train()

elif args.command == 'test':
    # Inference should use the config with parameters that are used in training
    cfg.MODEL.WEIGHTS = args.weights # os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    for d in random.sample(dataset_dicts_test, 3):
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
        #img_out_resized = ResizeWithAspectRatio(img_out, width=800)
        cv2.imshow("rand_name", img_out)  # out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Evaluating model performance on TEST set.
    evaluator = COCOEvaluator("arc_test", ("bbox", "segm"), True, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, 'arc_test')

    inference_on_dataset(trainer.model, val_loader, evaluator)
    # another equivalent way to evaluate the model is to use `trainer.test`


elif args.command == 'inference':
    print(args.image_path)
    # Inference should use the config with parameters that are used in training
    cfg.MODEL.WEIGHTS = args.weights  # path to the weights for inference.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    # Visualize the Predictions
    im = cv2.imread(args.image_path)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=taco_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW,
                   #instance_mode=ColorMode.SEGMENTATION,
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img_out = out.get_image()#[:, :, ::-1]

    # Converting to RGB (fixing for display)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    # adding a timestamp to testing
    time_out = get_timestamp()

    cv2.imwrite('./img_out/prediction'+time_out+"_at_" + str(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST) +'.jpg', img_out)
    # to visualize output uncomment below
    # cv2.imshow(out.get_image()[:, :, ::-1])

    # # Give information about predicted classes and boxes
    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)

elif args.command == "infer_mask":
    # Inference should use the config with parameters that are used in training
    cfg.MODEL.WEIGHTS = args.weights  # path to the weights for inference.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    # adding a timestamp to the images
    time_out = get_timestamp()

    # # saving the mask for a random image
    for d in random.sample(dataset_dicts_test, 1):
        print(d["file_name"])
        assert os.path.isfile(d["file_name"]), "Image not loaded correctly!"
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=taco_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)

        # image too large to display - resize down to fit in the screen
        img_new = out.get_image()[:, :, ::-1]
        img_resized = ResizeWithAspectRatio(img_new, width=800)
        cv2.imshow("rand_name", img_resized)  # out.get_image()[:, :, ::-1])
        cv2.imwrite('./img_out/mask_' + time_out + '.jpg', img_new)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # # Display the prediction on the same images
    outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(img[:, :, ::-1],
                   metadata=taco_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW,
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img_out = out.get_image()  # [:, :, ::-1]
    # Converting to RGB (fixing for display)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./img_out/pred_' + time_out + '.jpg',img_out)

    # # Give information about predicted classes and boxes
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

elif args.command == "predict_coco":
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    im = cv2.imread(args.image_path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img_out = out.get_image()  # [:, :, ::-1]

    # Converting to RGB (fixing for display)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    # adding a timestamp to testing
    time_out = get_timestamp()
    cv2.imwrite('./img_out/prediction' + time_out + '.jpg', img_out)

    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
