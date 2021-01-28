## ARCNET - Autonomous Trash Classifier

The arcnet is a custom network developed to automatically detect and label riverine waste. It builds upon the Detectron2 developed by Facebook research. 

#### Dependencies
This implementation used the following packages and versions. 
    
    - CUDA: 11.0
    - Nvidia Driver 450.80.02
    - Pytorch 1.7.0
    - Python 3.8.5
    - Ubuntu 18.04.5 LTS
    
Detectron2 system can be installed directly from [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). 

The model was custom trained using the [TACO](http://tacodataset.org/) dataset, and the classes were remapped to:
- "name": other litter, "id": 1 
- "name": plastic litter, "id": 2


Procedure:

This implementation uses adapts code written by TACO's developer, P. Proenza, to split and remap the plastic debris dataset. 

1. Split the dataset into ´train´ and ´test´. This can be done by calling the ´split_dataset.py´ function. Required arguments are 
    - --dataset_dir: path to the dataset's ’annotations.json’ file. 
2. Remap the dataset to the desired classes. This can be done by setting the arguments in the ’train_taco.py’ script.  
3. Call the train_taco.py function to train the custom dataset using TACO  

Procedure for replication:
-
1. Clone this repository to your local machine
2. Download the TACO dataset data by calling:

    `python download.py`

3. Create a custom dataset by remapping classes to desired map us the remap_classses.py script. 

    `python remap_classes.py --class_map maps/map_to_2.csv `

4. Create a K-Fold Dataset for Training and Validation. 
    
    `TODO: add code for dataset split`
    
5. Download the ARC dataset for Testing 
    
    TODO: Include information about how to download dataset from segments.ai
    
6. Train Mask R-CNN Model 


First testing: (poor results)

    - LR: 0.000125
    - Iterations: 300
    - Workers: 2

Second testing: (better results)

    - LR: 0.0025
    - Iterations: 2000
    - Workers: 2
    - img per batch: 2
    - ROI Heads batch size per image: 264
    
 