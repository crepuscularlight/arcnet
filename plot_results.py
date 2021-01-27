import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

# Parsing global arguments
parser = argparse.ArgumentParser(description='Custom implementation of Detectron2 using the TACO dataset.')
#parser.add_argument('--data_path', required=True, default='./output/metrics.json',  metavar="/path/file.json", help='Data to Metrics')
#parser.add_argument('--metrics_left', required=True, default='[metrics]', metavar="metrics to plot",  nargs='+', help='metrics to plot as an array')
#parser.add_argument('--metrics_right', required=False, default='[metrics]', metavar="metrics to plot",  nargs='+', help='metrics to plot as an array')
args = parser.parse_args()

# # # # Configurations # # #
color_base = 'black'
color_APsml = 'tab:orange'
color_AP = 'tab:blue'
lr = "lr 0.0025"
#freeze = "freeze at 2"
experiment_folder = './output/2_class_plastic_other/lr00025_2class/lr00025_cocoeval20_iter800_freeze'

# Freeze points:
exp_fr1 = load_json_arr(experiment_folder+'1/metrics.json')
exp_fr2 = load_json_arr(experiment_folder+'2/metrics.json')
exp_fr3 = load_json_arr(experiment_folder+'3/metrics.json')
exp_fr4 = load_json_arr(experiment_folder+'4/metrics.json')
exp_fr5 = load_json_arr(experiment_folder+'5/metrics.json')


#experiment_metrics = load_json_arr(args.data_path)
# # # # # # # # # # # # # #

# Plotting bbox AP and Segm AP metrics.
fig, ax1 = plt.subplots()
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Average Precision')


ax1.plot(
    [x['iteration'] for x in exp_fr1 if 'bbox/AP50' in x],
    [x['bbox/AP50'] for x in exp_fr1 if 'bbox/AP50' in x], color=color_base, label="Freeze 1")
ax1.plot(
    [x['iteration'] for x in exp_fr2 if 'bbox/AP50' in x],
    [x['bbox/AP50'] for x in exp_fr2 if 'bbox/AP50' in x], color=color_AP, label="Freeze 2", linestyle="dashed")
ax1.plot(
    [x['iteration'] for x in exp_fr3 if 'bbox/AP50' in x],
    [x['bbox/AP50'] for x in exp_fr3 if 'bbox/AP50' in x], color=color_AP, label="Freeze 3", linestyle="-.")
ax1.plot(
    [x['iteration'] for x in exp_fr4 if 'bbox/AP50' in x],
    [x['bbox/AP50'] for x in exp_fr4 if 'bbox/AP50' in x], color=color_AP, label="Freeze 4",  linestyle="dotted")
ax1.plot(
    [x['iteration'] for x in exp_fr5 if 'bbox/AP50' in x],
    [x['bbox/AP50'] for x in exp_fr5 if 'bbox/AP50' in x], color=color_AP, label="Freeze 5")

ax1.tick_params(axis='y')
plt.legend(loc='best')
plt.title(f"MRCNN Metrics - Bounding Box AP - TACO at {lr}")
plt.show()


# Plotting segmentation metrics
fig, ax1 = plt.subplots()
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')

ax1.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], color=color_base, label="Total Loss")

ax1.tick_params(axis='y')
plt.legend(loc='upper left')

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('AP')
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'segm/AP' in x],
    [x['segm/AP'] for x in experiment_metrics if 'segm/AP' in x], color=color_AP, label="Segmentation AP", linestyle="dashed")
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'segm/AP50' in x],
    [x['segm/AP50'] for x in experiment_metrics if 'segm/AP50' in x], color=color_AP, label="Segmentation AP@.50")
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/AP75' in x],
    [x['segm/AP75'] for x in experiment_metrics if 'bbox/AP75' in x], color=color_AP, label="Segmentation AP@.75", linestyle="-.")

# Plotting size dependent metrics
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/APs' in x],
    [x['segm/APs'] for x in experiment_metrics if 'bbox/APs' in x], color=color_APsml, label="Segmentation APs")
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/APm' in x],
    [x['segm/APm'] for x in experiment_metrics if 'bbox/APs' in x], color=color_APsml, label="Segmentation APm", linestyle="-.")
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/APl' in x],
    [x['segm/APl'] for x in experiment_metrics if 'bbox/APl' in x], color=color_APsml, label="Segmentation APl", linestyle="dotted")

ax2.tick_params(axis='y')
plt.legend(loc='upper right')
plt.title(f"MRCNN Metrics - Segmentation AP - TACO at {lr}")
plt.show()


# Plotting Accuracy, False Positive and False Negative Metrics
fig, ax1 = plt.subplots()
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')

ax1.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], color=color_base, label="Total Loss")

ax1.tick_params(axis='y')
plt.legend(loc='upper left')

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Percent')
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'mask_rcnn/accuracy' in x],
    [x['mask_rcnn/accuracy'] for x in experiment_metrics if 'mask_rcnn/accuracy' in x], color=color_AP, label="Mask R-CNN Accuracy", linestyle="dashed")
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'mask_rcnn/false_negative' in x],
    [x['mask_rcnn/false_negative'] for x in experiment_metrics if 'mask_rcnn/false_negative' in x], color=color_APsml, label="Mask R-CNN False Negative")
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'mask_rcnn/false_positive' in x],
    [x['mask_rcnn/false_positive'] for x in experiment_metrics if 'mask_rcnn/false_positive' in x], color="tab:red", label="Mask R-CNN False Positive", linestyle="-.")


ax2.tick_params(axis='y')
plt.legend(loc='best')
plt.title(f"MRCNN Performance Metrics - {lr} - {freeze}")
plt.show()