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

