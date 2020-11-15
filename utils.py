# Utilities developed for custom detector using Detectron and TACO dataset.
import cv2
import copy
import os
import csv
import json
from pycocotools.coco import COCO

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def replace_dataset_classes(dataset, class_map):
    """ Replaces classes of dataset based on a dictionary"""
    class_new_names = list(set(class_map.values()))
    class_new_names.sort()
    class_originals = copy.deepcopy(dataset['categories'])
    dataset['categories'] = []
    class_ids_map = {}  # map from old id to new id

    # Assign background id 0
    has_background = False
    if 'Background' in class_new_names:
        if class_new_names.index('Background') != 0:
            class_new_names.remove('Background')
            class_new_names.insert(0, 'Background')
        has_background = True

    # Replace categories
    for id_new, class_new_name in enumerate(class_new_names):

        # Make sure id:0 is reserved for background
        id_rectified = id_new
        if not has_background:
            id_rectified += 1

        category = {
            'supercategory': '',
            'id': id_rectified,  # Background has id=0
            'name': class_new_name,
        }
        dataset['categories'].append(category)
        # Map class names
        for class_original in class_originals:
            if class_map[class_original['name']] == class_new_name:
                class_ids_map[class_original['id']] = id_rectified

    # Update annotations category id tag
    for ann in dataset['annotations']:
        ann['category_id'] = class_ids_map[ann['category_id']]

def get_taco_dicts(img_dir):
    json_file = os.path.join(img_dir, "annotations.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

"""
for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")
"""

if __name__ == "__main__":
    dataset_dir = "./data/"
    # Read map of target classes
    class_map = {}
    map_to_one_class = {}
    with open("./taco_config/map_1.csv") as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]: row[1] for row in reader}
        map_to_one_class = {c: 'Litter' for c in class_map}

    ann_filepath = os.path.join(dataset_dir, 'annotations.json')
    dataset = json.load(open(ann_filepath, 'r'))

    """
    # Some classes may be assigned background to remove them from the dataset
    replace_dataset_classes(dataset, class_map)

    # Creating a new dataset with the mapped classes
    taco_alla_coco = COCO()
    taco_alla_coco.dataset = dataset
    taco_alla_coco.createIndex()

    # Add images and classes except Background
    # Definitely not the most efficient way
    image_ids = []
    background_id = -1
    class_ids = sorted(taco_alla_coco.getCatIds())
    for i in class_ids:
        class_name = taco_alla_coco.loadCats(i)[0]["name"]
        if class_name != 'Background':
            self.add_class("taco", i, class_name)
            image_ids.extend(list(taco_alla_coco.getImgIds(catIds=i)))
        else:
            background_id = i
    image_ids = list(set(image_ids))

    if background_id > -1:
        class_ids.remove(background_id)

    print('Number of images used:', len(image_ids))

    # Add images
    for i in image_ids:
        self.add_image(
            "taco", image_id=i,
            path=os.path.join(dataset_dir, taco_alla_coco.imgs[i]['file_name']),
            width=taco_alla_coco.imgs[i]["width"],
            height=taco_alla_coco.imgs[i]["height"],
            annotations=taco_alla_coco.loadAnns(taco_alla_coco.getAnnIds(
                imgIds=[i], catIds=class_ids, iscrowd=None)))
    if return_taco:
        return taco_alla_coco

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=taco_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("rand_name", vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    json_file = "../TACO/data/annotations.json"
    with open(json_file) as f:
        img_anns = json.load(f)

    for idx, v in enumerate(img_anns.values()):
        record = {}
"""

    # print(img_anns["categories"])

