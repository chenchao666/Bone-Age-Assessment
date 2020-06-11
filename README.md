# Attention-Guided Discriminative Region Localization for Bone Age Assessment
<div align=center><img src="https://github.com/chenchao666/Bone-Age-Assessment/blob/master/img/img1.png" width="750" /></div>

* This repository contains code for our paper **Attention-Guided Discriminative Region Localization for Bone Age Assessment** [Download Paper](https://arxiv.org/abs/2006.00202)
* If you have any question about our paper or code, please don't hesitate to contact with me ahucomputer@126.com, we will update our repository accordingly

## Setup
* **Dataset** The code as well as the dataset can be downloaded here [HoMM in MNIST](https://drive.google.com/open?id=167tVIBI2dVa0D18i6CiM-hicFJ3DJFzX) 

* **requirements** Python==3.6, tensorflow==1.9, keras = 2.1.6, opencv.

## Training
* Step 1: Generate .npy Data with data_utils.py
* Step 2: Run main_classification.py to train classification model, the Attention Map and Heatmap will be saved according to the given path.
* Step 3: Generate the Heatmap for the Hand Reigon, Region-1 and Region-2 one-by-one
* Step 4: Run data/crop_patches.py to crop the local patches for Hand, Region-1 and Region-2 according to the heatmap one-by-one.
* Step 5: Run main_aggregation.py to aggragate different local patches for BAA
* You can also run main_regression.py to get the BAA performance by using one local patch.

## Results
<div align=center><img src="https://github.com/chenchao666/Bone-Age-Assessment/blob/master/img/img2.png" width="750" /></div>
<div align=center><img src="https://github.com/chenchao666/Bone-Age-Assessment/blob/master/img/img3.png" width="750" /></div>

## Citation
* If you find it helpful for you, please cite our paper
```
@inproceedings{chen2020HoMM,
  title={HoMM: Higher-order Moment Matching for Unsupervised Domain Adaptation},
  author={Chao Chen, Zhihang Fu, Zhihong Chen, Sheng Jin, Zhaowei Cheng, Xinyu Jin, Xian-Sheng Hua},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  year={2020}
}
