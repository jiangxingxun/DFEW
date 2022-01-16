# DFEW

*Xingxun Jiang, Yuan Zong, Wenming Zheng, Chuangao Tang, Wanchuang Xia, Cheng Lu, Jiateng Liu "[DFEW: A Large-Scale Database for Recognizing Dynamic Facial Expressions in the Wild](https://dl.acm.org/doi/10.1145/3394171.3413620)". ACM MM'20*

## Requirements
- Python == 3.6.0
- PyTorch == 1.8.0
- Torchvision == 0.8.0

## Training 
- Step 1: download the single-labeled samples of [DFEW](https://dfew-dataset.github.io/) dataset, and make sure it has the structure like the following:

```txt
/data/jiangxingxun/Data/DFEW/data_affine/single_label/
                                                      data/
                                                           00001/
                                                                 00001_00001.jpg
                                                                 ...
                                                                 00001_00144.jpg
                                                           16372/
                                                                 16372_00001.jpg
                                                                 ...
                                                                 16372_00039.jpg
                                                      label/
                                                            single_trainset_1.csv
                                                            ...
                                                            single_trainset_5.csv
                                                            single_testset_1.csv
                                                            ...
                                                            single_testset_5.csv
[Note]: 1:Happy 2:Sad 3:Neutral 4:Angry 5:Surprise 6:Disgust 7:Fear
```

- Step 2: run ```run.sh```

## Pretrained Models
- We provided the model's pretrained weights of each fold under 5-fold cross-validation protocol.

- To get the better metrics, we provided the models trained by extracted frames from videos, other than those acquired by the [Time Interpolation Method](https://ieeexplore.ieee.org/document/6601598) with matlab [code](https://github.com/jiangxingxun/TIM_oulu).

- We take two metrics to evaluate models, i.e., WAR and UAR. WAR is the weighted average recall, i.e., accuracy; and UAR is the unweighted average recall, i.e., the accuracy per class divided by the number of classes without considerations of instances per class.


- you can download model by wget command like ```wget https://aip.seu.edu.cn/download/pretrained_model/DFEW/scratch/<model_name>/```, where <model_name> can be replaced with mc3_18, r3d_18, i3d ...
[^_^]: or you can download the pretrained weights from [Baidu Disk](https://pan.baidu.com/s/1ys6bH3T3e-TrBwWye73PPQ) with Access Code (8azt) or [Google Driver](https://drive.google.com/drive/folders/11gVqH4WULvY_Gp-yHJTERdrzHukzsoIh?usp=sharing). >




|model_name|ref|WAR(%)|UAR(%)|Note|
|:---|:---|:---:|:---:|:---|
|r3d_18|[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf)|55.70|45.11|-|
|mc3_18|[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf)|57.02|46.50|-|
|i3d|[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf)|59.24|47.61|I3D-RGB|

## Citation
```txt
@inproceedings{jiang2020dfew,
  title={Dfew: A large-scale database for recognizing dynamic facial expressions in the wild},
  author={Jiang, Xingxun and Zong, Yuan and Zheng, Wenming and Tang, Chuangao and Xia, Wanchuang and Lu, Cheng and Liu, Jiateng},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2881--2889},
  year={2020}
}
```

