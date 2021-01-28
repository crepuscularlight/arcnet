from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import os

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('coco_eval', exist_ok=True)
            output_folder ='coco_eval'

        return COCOEvaluator(dataset_name, ("bbox", "segm"), False, output_folder)

