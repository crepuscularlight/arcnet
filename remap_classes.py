"""
Author: Fidel Esquivel Estay
Year: 2020
This script takes a JSON dataset in the COCO format and a dictionary of class maps and outputs
a remapped JSON file with the new dataset for training
------------------------------------------------------

Every COCO style dataset stores the information of the classes in the form of list of dictionaries. Therefore, to change
the class, we need to modify the information about every annotation. For this, it is also necessary to change the list
of categories, super categories, and other relevant data.

The categories have the following format.
dataset {
- List of Annotation dictionaries
    - [{each annotation as dictionary}, {},{},...]
        -{'categorie1':'value', 'cat2':'val', ...}

- List of categories -  dictionaries
    - [{catetgorie1 dictionary}, {categorie2 dict},{},...]

- other lists of dictionaries
    -[{},{},...]

} #end of dataset dict.

Dictionary keys for TACO
dict_keys(['info', 'images', 'annotations', 'scene_annotations', 'licenses', 'categories', 'scene_categories'])

NOTE: For TACO, scene categories, and annotations are independent of the categories, can be modified separately.
"""

import json
import os.path
import json
import argparse
import numpy as np
import random
import datetime as dt
import copy
import csv

def remap_classes(dataset, class_map):
    """ Replaces classes of dataset based on a dictionary"""
    class_new_names = list(set(class_map.values()))
    class_new_names.sort() # NOTE sort() is a NoneType return method, it sorts the list without outputting new vars
    class_originals = copy.deepcopy(dataset['categories'])
    dataset['categories'] = []  # removing all dependencies
    class_ids_map = {}  # map from old id to new id

    # Check whether the category has background or not, assign index 0. Useful for panoptic segmentation.
    has_background = False
    if 'Background' in class_new_names:
        # Check whether the backgroun category has index zero.
        if class_new_names.index('Background') != 0:
            class_new_names.remove('Background')
            class_new_names.insert(0, 'Background')
        has_background = True

    # Catching duplicates - TACO had duplicates for id 4040 and 309. Re-id'd
    id_ann_all = []
    id_ann_repeated = []
    for index_old, ann_old in enumerate(dataset['annotations']):
        if ann_old['id'] in id_ann_all:
            # if found a duplicate, re-id at the end
            id_ann_repeated.append(ann_old['id'])
            ann_old['id'] = len(dataset['annotations'])+len(id_ann_repeated)-1
        else:
            id_ann_all.append(ann_old['id'])
    print(f'Found {len(id_ann_repeated)} annotations repeated.'
          f'\nPlease double check input file, annotation id(s) {id_ann_repeated} are duplicated!\n')

    # Replace categories, iterating through every class name
    for id_new, class_new_name in enumerate(class_new_names):
        # Make sure id:0 is reserved for background
        id_rectified = id_new
        if not has_background:
            id_rectified += 1

        # Creating new category dictionary, using new category ID and the new class name
        category = {
            'supercategory': '',
            'id': id_rectified,  # Background has id=0
            'name': class_new_name,
        }
        dataset['categories'].append(category)  # assigning new categories

        # Map class names
        for class_original in class_originals:
            # If the new class exists in the value of the class map dict, create new class id
            if class_map[class_original['name']] == class_new_name:
                class_ids_map[class_original['id']] = id_rectified

    # Update annotations category id tag
    for ann in dataset['annotations']:
        ann['category_id'] = class_ids_map[ann['category_id']]

    # Saving the newly created file as a JSON file
    num_classes = str(len(class_new_names))
    ann_out_path = './data' + '/' + 'annotations_'+ 'map_to_' + num_classes +'.json'
    with open(ann_out_path, 'w+') as f:
        f.write(json.dumps(dataset))

    # return path to new file, for loading somewhere else.
    return str(os.path.abspath(ann_out_path))
if __name__ == '__main__':
    """
    remap classes of custom dataset 
    Args:
        - class map-> path to csv file containing dictionary of mapped classes (original,new)
        - annotaions -> path to json file containing original annotations in COCO format
    Returns:
        - JSON file with dataset. 
    """
    # Parsing global arguments
    parser = argparse.ArgumentParser(description='Class remapper for TACO dataset')
    parser.add_argument('--class_map', required=False, metavar="/dir/map.csv", help='Target classes')
    parser.add_argument('--ann_dir', required=False, metavar='/path_to_anns/annotation.json', help='path to .json')
    args = parser.parse_args()

    # Load class map
    if args.class_map is not None:
        class_map_path = args.class_map
    else:
        class_map_path = "./maps/map_to_2.csv"  # Default is set to two classes: [plastic litter, other litter]

    # Reading map and creating map dictionary
    class_map = {}
    assert os.path.isfile(class_map_path), "Class map file not found!"
    with open(class_map_path) as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]: row[1] for row in reader}

    # Reading the annotations
    if args.ann_dir is not None:
        ann_input_path = args.ann_dir
    else:
        ann_input_path = './data/annotations.json'  # Default is set to two classes: [plastic litter, other litter]

    # Creating the json file dataset
    assert os.path.isfile(ann_input_path), "Annotations file not found!"
    with open(ann_input_path, 'r') as f:
        dataset = json.loads(f.read())

    # Creating the new mapped dataset json file. Output of this function is the creation of a new file.
    remapped_json_path = remap_classes(dataset, class_map)
    print(f'New dataset created.\nSaved to file {remapped_json_path}')