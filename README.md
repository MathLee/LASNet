# LASNet 
  This project provides the code and results for 'RGB-T Semantic Segmentation with Location, Activation, and Sharpening', IEEE TCSVT, 2022. [IEEE link](https://ieeexplore.ieee.org/document/9900351)
  
# Requirements
  python 3.7/3.8 + pytorch 1.9.0 (based on [EGFNet](https://github.com/ShaohuaDong2021/EGFNet))
   
   
# Segmentation maps
   We provide segmentation maps on MFNet dataset and PST900 dataset under './model/'.
   
# Training
1. Install 'apex'.
  
2. Download [MFNet dataset](https://pan.baidu.com/s/1NHGazP7pwgEM47SP_ljJPg) (code: 3b9o) or [PST900 dataset](https://pan.baidu.com/s/13xgwFfUbu8zNvkwJq2Ggug) (code: mp2h).
  
3. Run train_LASNet.py (default to MFNet Dataset).


# Pre-trained model and testing
1. Download the following pre-trained model and put it under './model/'.
[model_MFNet.pth](https://pan.baidu.com/s/1dWCbTl274nzgdHGOsJkK_Q) (code: 5th1)   [model_PST900.pth](https://pan.baidu.com/s/1zQif2_8LTG5R7aabQOXjrA) (code: okdq)

2. Rename the name of the pre-trained model to 'model.pth', and then run test_LASNet.py (default to MFNet Dataset).
  
  
# Citation
        @ARTICLE{Li_2022_LASNet,
                author = {Gongyang Li and Yike Wang and Zhi Liu and Xinpeng Zhang and Dan Zeng},
                title = {RGB-T Semantic Segmentation with Location, Activation, and Sharpening},
                journal = {IEEE Transactions on Circuits and Systems for Video Technology},
                year = {2022},
                doi = {10.1109/TCSVT.2022.3208833},
                }
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at lllmiemie@163.com or ligongyang@shu.edu.cn.
