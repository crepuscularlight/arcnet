## ARCNET - Autonomous Trash Classifier

The arcnet is a custom network developed to automatically detect and label riverine waste. It builds upon the Detectron2 developed by Facebook research. 

#### Dependencies
This implementation used the following packages and versions. 
    
    - CUDA: 11.0
    - Nvidia Driver 450
    - Pytorch 1.7.0
    - Python 3.8.5
    - Ubuntu 18.04.5 LTS
    
Detectron sytem can be installed directly from [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). 

The model was custom trained using the [TACO](http://tacodataset.org/) dataset, and the classes were defined to be:
- Litter
- Background