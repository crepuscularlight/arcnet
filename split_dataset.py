"""
This script takes a dataset in coco format (json file) and returns three datasets with train-test-val splits.
To run (for ARC application example)
    python split_dataset.py --dataset_dir data --out_name map_2  --nr_trials 1
"""

import os.path
import json
import argparse
import numpy as np
import random
import datetime as dt
import copy

parser = argparse.ArgumentParser(description='User args')
parser.add_argument('--dataset_dir', required=True, help='Path to original dataset annotations')
parser.add_argument('--test_percentage', type=int, default=10, required=False, help='Percentage of images used for the testing set')
parser.add_argument('--val_percentage', type=int, default=10, required=False, help='Percentage of images used for the validation set')
parser.add_argument('--out_name', type=str, default='', required=False, help='renaming output file')
parser.add_argument('--nr_trials', type=int, default=1, required=False, help='Number of splits')
args = parser.parse_args()

# TODO remove hardcoded annotation input path (possibly combine with remapping function)
ann_input_path = args.dataset_dir + '/' + 'annotations_map_2.json'

# Load annotations
with open(ann_input_path, 'r') as f:
    dataset = json.loads(f.read())

# Extract annotations
anns = dataset['annotations']
scene_anns = dataset['scene_annotations']
imgs = dataset['images']
nr_images = len(imgs)

nr_testing_images = int(nr_images*args.test_percentage*0.01+0.5)
nr_nontraining_images = int(nr_images*(args.test_percentage+args.val_percentage)*0.01+0.5)


for i in range(args.nr_trials):
    random.shuffle(imgs)

    # Add new datasets
    train_set = {'info': dataset['info'],
                 'images': [],
                 'annotations': [],
                 'scene_annotations': [],
                 'licenses': [],
                 'categories': dataset['categories'],
                 'scene_categories': dataset['scene_categories']}

    val_set = copy.deepcopy(train_set)
    test_set = copy.deepcopy(train_set)

    test_set['images'] = imgs[0:nr_testing_images]
    val_set['images'] = imgs[nr_testing_images:nr_nontraining_images]
    train_set['images'] = imgs[nr_nontraining_images:nr_images]

    # Aux Image Ids to split annotations
    test_img_ids, val_img_ids, train_img_ids = [],[],[]
    for img in test_set['images']:
        test_img_ids.append(img['id'])

    for img in val_set['images']:
        val_img_ids.append(img['id'])

    for img in train_set['images']:
        train_img_ids.append(img['id'])

    # Split instance annotations
    for ann in anns:
        if ann['image_id'] in test_img_ids:
            test_set['annotations'].append(ann)
        elif ann['image_id'] in val_img_ids:
            val_set['annotations'].append(ann)
        elif ann['image_id'] in train_img_ids:
            train_set['annotations'].append(ann)

    # Split scene tags
    for ann in scene_anns:
        if ann['image_id'] in test_img_ids:
            test_set['scene_annotations'].append(ann)
        elif ann['image_id'] in val_img_ids:
            val_set['scene_annotations'].append(ann)
        elif ann['image_id'] in train_img_ids:
            train_set['scene_annotations'].append(ann)

    # Write dataset splits
    ann_train_out_path = args.dataset_dir + '/' + 'annotations_' + str(i) + "_" + args.out_name + '_train.json'
    ann_val_out_path   = args.dataset_dir + '/' + 'annotations_' + str(i) + "_" + args.out_name + '_val.json'
    ann_test_out_path  = args.dataset_dir + '/' + 'annotations_' + str(i) + "_" + args.out_name + '_test.json'

    with open(ann_train_out_path, 'w+') as f:
        f.write(json.dumps(train_set))

    with open(ann_val_out_path, 'w+') as f:
        f.write(json.dumps(val_set))

    with open(ann_test_out_path, 'w+') as f:
        f.write(json.dumps(test_set))


