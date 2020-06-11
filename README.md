# Attention-Guided Discriminative Region Localization for Bone Age Assessment
<div align=center><img src="https://github.com/chenchao666/Bone-Age-Assessment/blob/master/img/img1.png" width="750" /></div>
* This repository contains code for our paper ** Attention-Guided Discriminative Region Localization for Bone Age Assessment** [Download Paper](https://arxiv.org/abs/2006.00202)
* If you have any question about our paper or code, please don't hesitate to contact with me ahucomputer@126.com, we will update our repository accordingly

## Setup
* **Dataset** The code as well as the dataset can be downloaded here [HoMM in MNIST](https://drive.google.com/open?id=167tVIBI2dVa0D18i6CiM-hicFJ3DJFzX) 

* **requirements** Python==2.7, tensorflow==1.9, opencv

## Training
* **MNIST** You can run **TrainLenet.py** in HoMM-mnist.
* **Office&Office-Home** You can run **finetune.py** in HoMM_office/resnet/.
* We have provide four functions **HoMM3**, **HoMM4**, **HoMM** and **KHoMM** conresponding to the third-order HoMM, fourth-order HoMM, Arbitrary-order moment matching, and Kernel HoMM.

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
