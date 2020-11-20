## ARCNET - Autonomous Trash Classifier

The arcnet is a custom network developed to automatically detect and label riverine waste. It builds upon the Detectron2 developed by Facebook research. This is part of the Autonomous River Cleanup project, arc.ethz.ch.

#### Dependencies
This implementation used the following packages and versions. 
    
    - CUDA: 11.0
    - Nvidia Driver 450
    - Pytorch 1.7.0
    - Python 3.8.5
    - Ubuntu 18.04.5 LTS
Additional dependencies can be installed directly by running:
    
    - pip3 install -r requirements.txt

Detectron2 can be installed directly from [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). 

#### Dataset Overview 
The model was custom trained using the [TACO](http://tacodataset.org/) dataset. The original dataset contains 60 categories and 28 supercategories.
For this implementation, classes were remapped into just two categories:
- Plastic Litter
- Other Litter

There are 1500 distinct images, and a total of 4600+ annotations. 


Procedure:
-

This implementation uses code developed by TACO's developer, P. Proenza, to split and remap the plastic debris dataset. 

NOTES on Implementation:

1. The dataset can be downloaded directly from flickr. Run the download.py function in this repository to download the TACO dataset to your local machine. 


    # Step 1:   Remap output to the desired number of classes. Choose a map (dictionary) or create
    #           your own and place it in the folder 'maps'.

    #   - - Run the remapping - -
    #           python remap_classes --class_map <path_to_map/file.csv> --ann_dir <path_to_annotations/file.json>

    # Step 2:   Split dataset into train-test splits, k-times for k-fold cross validation.

    #   - - Split the dataset - -
    #           python split_dataset.py --nr_trials <K_folds> --out_name <name of file> --dataset_dir <path_to_data>

    # To train the model:
    #    Template: python arcnet_main.py --data_dir <path_to_dataset/> train
    #    EXAMPLE:  python arcnet_main.py --class_map maps/map_to_2.csv --data_dir data train

    # To test the model:
    #    Template: python arcnet_main.py test

    # To try the model for inference an image
    #    TEMPLATE: python arcnet_main.py inference --image_path <path/to/test_image.jpg>
    #    EXAMPLE:  python arcnet_main.py inference --image_path img_test/test_img1.jpg

     # Check Tensorboard for model training validation information.
     tensorboard --logdir ./output/


Results 
-

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

Third iteration:     
 