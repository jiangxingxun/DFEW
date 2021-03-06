import os
import time
import yaml

from libs import DFEW_Dataset
from libs import model_metrics

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import numpy as np
import random

import sklearn.metrics
from tensorboardX import SummaryWriter

import pdb

class Performer():
    def __init__(self, args):
        '''
        Initilization: 1. save args,  2. GPU memory,  3. prepare data & dataloader,   4. model,   5. optimizer
        ------------------
        Return: None
        '''
        self.args = args
        self.save_args()
        self.seed = self.load_torch_optimize()
        self.make_dataloader(loader_types=["train", "test"])
        self.load_model()
        self.load_optim()


    def save_args(self):
        '''
        Save parameters: from ./config/xx.yaml into workdir
        ------------------
        Return: None
        '''
        arg_dict = vars(self.args)
        timestamp = "{time:s}".format(time=time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        self.work_dir = os.path.join(self.args.work_dir, timestamp)

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        yaml_name = "{pth}/config.yaml".format(pth=self.work_dir)
        with open(yaml_name, "w") as f:
            yaml.dump(arg_dict, f)


    def load_torch_optimize(self):
        '''
        optimize GPU memory
        ------------------
        Return: None
        '''
        seed = int(time.time())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if self.args.torch_optimize == True:
            torch.backends.cudnn.enabled   = True
            torch.backends.cudnn.benchmark = True

        return seed


    def make_dataloader(self, loader_types=[]):
        '''
        Prepare data and dataloader: train and test
        ------------------
        Return: None
        '''
        # data path
        if int(self.args.gpu_id) in [7, 8]:
            data_ori_path = "/data"

        self.args.data_root = os.path.join(data_ori_path, self.args.data_root)
        data_type = self.args.data_type

        # data and dataloader
        if "train" in loader_types:
            self.train_data       = DFEW_Dataset.DFEW_Dataset(args  = self.args,
                                                              phase = "train")
            self.train_dataloader = DataLoader(self.train_data,
                                               batch_size=self.args.batch_size, 
                                               shuffle=True,
                                               num_workers=self.args.num_workers,
                                               pin_memory=self.args.pin_memory)

        if "test" in loader_types:
            self.test_data        = DFEW_Dataset.DFEW_Dataset(args  = self.args,
                                                              phase = "test")
            self.test_dataloader  = DataLoader(self.test_data, 
                                               batch_size=self.args.batch_size, 
                                               shuffle=False,
                                               num_workers=self.args.num_workers,
                                               pin_memory=self.args.pin_memory)


    def load_model(self):
        '''
        models: models, initialize weights, to device
        ------------------
        Return: None
        '''
        # model & training loss
        if self.args.model_name == "r3d_18":         
            self.model = torchvision.models.video.resnet.r3d_18(pretrained=self.args.model_pretrain)
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.args.num_classes)

        # Initializing model's parameters
        if self.args.model_init == True:
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias,   0)
            string = "Initilize: 1. Conv2d & Linear with kaiming_normal_ 2. BatchNorm2d with constant"
            self.txt_log(string)

        # data parallel
        if self.args.Flag_mGPU_blocks == True:
            device_ids_list = [int(ele) for ele in self.args.List_mGPU_blocks]
            self.model = nn.DataParallel(self.model, device_ids=device_ids_list)
        
        self.model.cuda()

        if self.args.loss_type == "CEL":
            self.loss  = nn.CrossEntropyLoss()

        self.loss.cuda()
        

    
    def load_optim(self):
        '''
        Load: 1. opmitimizer  2. scheduler
        ------------------
        Return: None
        '''
        params = self.model.parameters()
        
        # Optimizer Type
        if self.args.optimizer == "Adam":
            self.optimizer = optim.Adam(params,
                                        lr=self.args.lr_init)
        if self.args.optimizer == "SGD":
            self.optimizer = optim.SGD(params,
                                   lr=self.args.lr_init)

        # Leaning strategy
        if self.args.lr_strategy == True:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9)



    def txt_log(self, string):
        '''
        Records to screen and .txt
        ------------------
        Return: None
        '''
        localtime = time.asctime(time.localtime(time.time()))
        string    = "[{localtime}] {string}".format(localtime=localtime, string=string)
        if self.args.txt_log_toScreen == True:
            print(string)
        if self.args.txt_log == True:
            with open("{}/log.txt".format(self.work_dir),"a") as f:
                print(string, file=f)



    def start_train_epochTest(self):
        '''
        Train models: test every epoch
        ------------------
        Return: None
        '''
        if self.args.tensorboard == True:
            cmt    = "__fold#{fold}_lr#{lr}_bt#{bt}".format(fold = self.args.fold_idx,
                                                            lr   = self.args.lr_init,
                                                            bt   = self.args.batch_size_tr)
            writer = SummaryWriter(comment=cmt)

        WAR_TEST_MAX, UAR_TEST_MAX = 0.0, 0.0
        for epo in range(0, self.args.num_epoch):
            # Train: [1. Param]
            pres_tr, trues_tr = [], []
            running_loss_tr   = 0.0

            # Train: [2. Start] 
            self.model.train()
            for idx_tr, (X_tr, y_tr_s) in enumerate(self.train_dataloader):
                step_tr   = epo*len(self.train_dataloader) + idx_tr + 1
                if self.args.y_start_from_zero == False:
                    y_tr_s  = y_tr_s - 1

                X_tr      = X_tr.type(torch.FloatTensor).cuda().requires_grad_()
                y_tr_s    = y_tr_s.type(torch.LongTensor).cuda()

                # Model and Loss
                if self.args.loss_type == "CEL":
                    out_tr    = self.model(X_tr)
                    loss_tr= self.loss(out_tr, y_tr_s)

                # Loss Backward Optimizer
                running_loss_tr += loss_tr.item() 
                loss_tr.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                _, pre_tr  = torch.max(out_tr, 1)
                pres_tr   += pre_tr.cpu().numpy().tolist()
                trues_tr  += y_tr_s.cpu().numpy().tolist()
                
                if self.args.tensorboard == True:  writer.add_scalar("loss_tr", loss_tr.data, step_tr)

            # Train: [3. Report]
            pres_tr, trues_tr = [[ele] for ele in pres_tr], [[ele] for ele in trues_tr]
            acc_tr = sklearn.metrics.accuracy_score(trues_tr, pres_tr)
            if self.args.tensorboard == True:  writer.add_scalar("train acc", acc_tr, step_tr)
            if self.args.txt_log     == True:  self.txt_log("[Train] epo:{epo}/{num_epoch}, batch_idx_tr:{idx_tr}/{len_train_dataloader}, running_loss_tr:{running_loss_tr:.2f}, train acc:{acc_tr:.2f}%".format(epo       = epo+1,
                                                                                                                                                                                                                 num_epoch = self.args.num_epoch,
                                                                                                                                                                                                                 idx_tr    = idx_tr+1,
                                                                                                                                                                                                                 len_train_dataloader = len(self.train_dataloader),
                                                                                                                                                                                                                 running_loss_tr      = running_loss_tr/len(self.train_dataloader),
                                                                                                                                                                                                                 acc_tr    = acc_tr*100)) 

            
            # Test: [1. Param]
            pres_te, trues_te = [], []

            # Test: [2. Start]
            self.model.eval()
            for idx_te, (X_te, y_te_s) in enumerate(self.test_dataloader):
                with torch.no_grad():
                    if self.args.y_start_from_zero == False:
                        y_te_s = y_te_s - 1

                    X_te      = X_te.type(torch.FloatTensor).cuda().requires_grad_()
                    y_te_s    = y_te_s.type(torch.LongTensor).cuda()

                    if self.args.loss_type in ["CEL"]:
                        out_te = self.model(X_te)

                    _, pre_te = torch.max(out_te, 1)
                    pres_te  += pre_te.cpu().numpy().tolist() 
                    trues_te += y_te_s.cpu().numpy().tolist()

            # Test: [3. Report]
            pres_te, trues_te = [[ele] for ele in pres_te], [[ele] for ele in trues_te]
            acc_te  = model_metrics.get_WAR(trues_te, pres_te)
            WAR_te  = acc_te
            cm      = sklearn.metrics.confusion_matrix(trues_te, pres_te)
            UAR_te  = model_metrics.get_UAR(trues_te, pres_te)
            
            # Save ACC
            #step = epo*len(self.train_dataloader)+idx_tr+1
            pth_fold = os.path.join(self.work_dir, "pth")
            if not os.path.exists(pth_fold):
                os.makedirs(pth_fold)
            pth_name = os.path.join(pth_fold,                         
                                "{model_name}_epo{epo}_WAR{WAR_te:.2f}_UAR{UAR_te:.2f}.pth".format(model_name = self.args.model_name,
                                                                                                     epo        = epo, 
                                                                                                     UAR_te     = UAR_te*100,
                                                                                                     WAR_te     = WAR_te*100))
            torch.save({
                'seed'                 : self.seed,
                'epo'                  : epo,
                'model_state_dict'     : self.model.state_dict(),
                'optimizer_state_dict' : self.optimizer.state_dict(),
                },
                pth_name)

            file_name = "{model_name}_fold{fold_idx}_epo{epo}_UAR{UAR_te:.2f}_WAR{WAR_te:.2f}.npz".format(model_name=self.args.model_name, fold_idx=self.args.fold_idx, epo=str(epo).zfill(3), UAR_te=UAR_te*100, WAR_te=WAR_te*100)
            np.savez(os.path.join(pth_fold, file_name), pres_te=pres_te, trues_te=trues_te)

                
                

            if self.args.tensorboard == True:  writer.add_scalar("test acc", UAR_te, step_tr)
            if self.args.tensorboard == True:  writer.add_scalar("max test UAR", UAR_TEST_MAX, step_tr)
            if self.args.txt_log     == True:  self.txt_log("[Test] epo:{epo}/{num_epoch}, batch_idx_tr:{idx_tr}/{len_train_dataloader}, WAR_te:{WAR_te:.2f}%, UAR_te:{UAR_te:.2f}%".format(epo          = epo+1,
                                                                                                                                                                                            num_epoch    = self.args.num_epoch,     
                                                                                                                                                                                            idx_tr       = idx_tr+1,
                                                                                                                                                                                            len_train_dataloader = len(self.train_dataloader),
                                                                                                                                                                                            UAR_te       = UAR_te*100,
                                                                                                                                                                                            WAR_te       = WAR_te*100))
            

    

    

    
    


    




