import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

import os
import pandas as pd
import glob
import random

import pdb


class DFEW_Dataset(Dataset):
    def __init__(self, args, phase):
        # Basic info
        self.args  = args
        self.phase = phase

        # File path
        label_df_path  = os.path.join(self.args.data_root,
                                      "label/{data_type}_{phase}set_{fold_idx}.csv".format(data_type=self.args.data_type, 
                                                                                           phase=self.phase, 
                                                                                           fold_idx=str(int(self.args.fold_idx))
                                                                                           )
                                      )
        label_df       = pd.read_csv(label_df_path)
        
        # Imgs & Labels
        self.names         = label_df['video_name']
        self.videos_path   = [ os.path.join(self.args.data_root,"data/{name}".format(name=str(ele).zfill(5))) 
                               for ele in self.names ]
        self.single_labels = torch.from_numpy(np.array(label_df['label']))

        # Transforms
        self.my_transforms_fun_dataAugment  = self.my_transforms_fun_dataAugment()
        self.my_transforms_te               = self.my_transforms_fun_te()


    def __len__(self):
        return len(self.single_labels)


    def __getitem__(self, index):
        imgs_per_video = glob.glob(self.videos_path[index]+'/*')
        imgs_per_video = sorted(imgs_per_video)
        imgs_idx = self.generate_index(nframe=self.args.nframe, 
                                       idx_start=0,
                                       idx_end=len(imgs_per_video)-1, 
                                       phase=self.phase, 
                                       isconsecutive=self.args.isconsecutive)
        data = torch.zeros(3, self.args.nframe, self.args.size_Resize_te, self.args.size_Resize_te)
        for i in range(self.args.nframe):
            img = Image.open(imgs_per_video[imgs_idx[i]])
            if self.phase == "train":
                if self.args.train_data_augment == True:
                    img = self.my_transforms_fun_dataAugment(img)
                else:
                    img = self.my_transforms_te(img)
            if self.phase == "test":
                img = self.my_transforms_te(img)
            data[:, i, :, :] = img 

        single_label = self.single_labels[index]

        return data, single_label

    def generate_index(self, nframe, idx_start, idx_end, phase, isconsecutive):
        if (idx_end-idx_start+1) < nframe:
            idx_list_tmp = list(range(idx_start, idx_end+1))
            idx_list = []
            for j in range(100):
                idx_list = idx_list + idx_list_tmp
                if len(idx_list) >=nframe:
                    break
            if isconsecutive == True:
                if self.phase == "train":
                    idx_s = random.randint(idx_start, idx_end-nframe)
                else:
                    idx_s = int(idx_end - nframe - idx_start)
                idx_tmp = list(range(idx_s, idx_s+nframe))
                idx = [idx_list[idx_tmp[jj]] for jj in range(len(idx_tmp))]
            if isconsecutive == False:
                if self.phase == "train":
                    idx_tmp = random.sample(range(len(idx_list)), nframe)
                    idx_tmp.sort()
                    idx = [idx_list[idx_tmp[jj]] for jj in range(len(idx_tmp))]
                else:
                    idx_tmp = np.linspace(0, len(idx_list)-1, nframe).astype(int)
                    idx = [idx_list[idx_tmp[jj]] for jj in range(len(idx_tmp))]

        if (idx_end-idx_start+1) >= nframe:
            if isconsecutive == True:
                if self.phase == "train":
                    idx_s = random.randint(idx_start, idx_end-nframe)
                else:
                    idx_s = int(idx_end - nframe - idx_start)
                idx = list(range(idx_s, idx_s + nframe))
            if isconsecutive == False:
                if self.phase == "train":
                    idx = random.sample(range(idx_start, idx_end+1), nframe)
                    idx.sort()
                else:
                    idx = np.linspace(idx_start, idx_end, nframe).astype(int)

        return idx


    def my_transforms_fun_dataAugment(self):
        my_img_transforms_list = []
        if self.args.Flag_RandomRotation:        my_img_transforms_list.append(transforms.RandomRotation(degrees=self.args.degree_RandomRotation))
        if self.args.Flag_CenterCrop:            my_img_transforms_list.append(transforms.CenterCrop(self.args.size_CenterCrop))
        if self.args.Flag_RandomResizedCrop:     my_img_transforms_list.append(transforms.RandomResizedCrop(self.args.size_RandomResizedCrop))
        if self.args.Flag_RandomHorizontalFlip:  my_img_transforms_list.append(transforms.RandomHorizontalFlip(p=self.args.prob_RandomHorizontalFlip))
        if self.args.Flag_RandomVerticalFlip:    my_img_transforms_list.append(transforms.RandomVerticalFlip(p=self.args.prob_RandomVerticalFlip))
        my_img_transforms_list.append(transforms.ToTensor())

        my_tensor_transforms_list = []
        if (self.args.model_pretrain == True) and (self.args.pretrained_weights == "ImageNet"): my_tensor_transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))) 
        if self.args.Flag_RamdomErasing :        my_tensor_transforms_list.append(transforms.RandomErasing(p=self.args.prob_RandomErasing))  

        my_transforms_list = my_img_transforms_list + my_tensor_transforms_list
        my_transforms      = transforms.Compose(my_transforms_list)

        return my_transforms


    def my_transforms_fun_te(self):
        my_img_transforms_list = []
        if self.args.Flag_Resize_te == True: my_img_transforms_list.append(transforms.Resize(self.args.size_Resize_te))
        my_img_transforms_list.append(transforms.ToTensor())

        my_tensor_transforms_list = []
        if (self.args.model_pretrain == True) and (self.args.pretrained_weights == "ImageNet"): my_tensor_transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))) 

        my_transforms_list = my_img_transforms_list + my_tensor_transforms_list
        my_transforms      = transforms.Compose(my_transforms_list)

        return my_transforms
