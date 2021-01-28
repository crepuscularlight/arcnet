import json
import matplotlib.pyplot as plt
import numpy as np

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

# # # # Configurations # # #
color_base = 'black'
color_APsml = 'tab:orange'
color_AP = 'tab:blue'
lr = "lr 0.025"
freeze = "freeze at 1"
experiment_folder = './output/1_class_litter/lr00025_1class/lr00025_freeze2_1class_ap50_iter1500'
experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
# # # # # # # # # # # # # #

# Plotting only Total Loss Metric together with Validation Loss
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'total_val_loss' in x],
    [x['total_val_loss'] for x in experiment_metrics if 'total_val_loss' in x])
plt.legend(['total loss', 'validation loss'], loc='upper right')
plt.title(f"Total loss and Validation Loss for MRCNN Trained on TACO - {lr}")
plt.xlabel("Iteration")
plt.ylabel("Total Loss")
plt.show()


# Plotting bbox AP and Segm AP mccetrics.
fig, ax1 = plt.subplots()
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')

ax1.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], color=color_base, label="Total Loss")
ax1.tick_params(axis='y')
plt.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.set_ylabel('AP')
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/AP' in x],
    [x['bbox/AP'] for x in experiment_metrics if 'bbox/AP' in x], color=color_AP, label="BBox AP", linestyle="dashed")
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/AP50' in x],
    [x['bbox/AP50'] for x in experiment_metrics if 'bbox/AP50' in x], color=color_AP, label="BBox AP@.50")

ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/AP75' in x],
    [x['bbox/AP75'] for x in experiment_metrics if 'bbox/AP75' in x], color=color_AP, label="BBox AP@.75", linestyle="-.")

# Debugging:
print([x['bbox/AP50'] for x in experiment_metrics if 'bbox/AP' in x])
# Plotting Size dependent metrics (APs, APm, APl)
"""ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/APs' in x],
    [x['bbox/APs'] for x in experiment_metrics if 'bbox/APs' in x], color=color_APsml, label="BBox APs")
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/APm' in x],
    [x['bbox/APm'] for x in experiment_metrics if 'bbox/APs' in x], color=color_APsml, label="BBox APm", linestyle="-.")
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/APl' in x],
    [x['bbox/APl'] for x in experiment_metrics if 'bbox/APl' in x], color=color_APsml, label="BBox APl", linestyle="dotted")
"""
ax2.tick_params(axis='y')
plt.legend(loc='upper right')
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
"""
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
"""
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