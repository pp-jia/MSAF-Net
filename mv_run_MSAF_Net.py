# -*- coding: utf-8 -*-
# @Time: 2024/9/2 1:54
# @Author: saku
import collections
import json
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader_MSAF_Net import *
from utils.Trainer_MSAF_Net import *
from model.MSAF_Net import *
import optuna


class Run():
    def __init__(self, config):
        self.dataset = config['dataset']
        self.mode = config['mode']
        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.early_stop = config['early_stop']
        self.device = config['device']
        self.path_ckp = config['path_ckp']
        self.path_tb = config['path_tb']
        self.inference_ckp = config['inference_ckp']
        self.config = config
        self.lr = config['lr']
        self.loss_weight = config['loss_weight']

    def get_dataloader(self, data_path):
        dataset = FakingRecipe_Dataset(data_path, self.dataset)
        collate_fn = collate_fn_FakeingRecipe
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        return dataloader

    def main(self):
        self.model = SimpleMultiView(self.config)

        if self.mode == 'train':
            if self.dataset == 'fakesv':
                data_split_dir = './data/FakeSV/data-split/temporal/'
                save_predict_result_path = './predict_result/FakeSV/'
            elif self.dataset == 'fakett':
                data_split_dir = './data/FakeTT/data-split/'
                save_predict_result_path = './predict_result/FakeTT/'

            train_data_path = data_split_dir + 'vid_time3_train.txt'
            test_data_path = data_split_dir + 'vid_time3_test.txt'
            val_data_path = data_split_dir + 'vid_time3_val.txt'

            data_load_time_start = time.time()
            train_dataloader = self.get_dataloader(train_data_path)
            test_dataloader = self.get_dataloader(test_data_path)
            val_dataloader = self.get_dataloader(val_data_path)
            dataloaders = dict(zip(['train', 'test', 'val'], [train_dataloader, test_dataloader, val_dataloader]))
            print('data load time: %.2f' % (time.time() - data_load_time_start))
            trainer = Trainer(model=self.model, device=self.device, lr=self.lr,
                              dataloaders=dataloaders,
                              epoches=self.epoches, model_name='MSAF_Net',
                              save_predict_result_path=save_predict_result_path,
                              loss_weight=self.loss_weight, early_stop=self.early_stop,
                              save_param_path=self.path_ckp + self.dataset + "/",
                              writer=SummaryWriter(self.path_tb + self.dataset + "/"))
            ckp_path = trainer.train()
        elif self.mode == 'inference_test':
            if self.dataset == 'fakesv':
                data_split_dir = './data/FakeSV/data-split/temporal/'
                save_predict_result_path = './predict_result/FakeSV/'
            elif self.dataset == 'fakett':
                data_split_dir = './data/FakeTT/data-split/'
                save_predict_result_path = './predict_result/FakeTT/'
            test_data_path = data_split_dir + 'vid_time3_test.txt'
            test_dataloader = self.get_dataloader(test_data_path)
            inferncer = Inferencer(model=self.model, device=self.device, model_name='MSAF_Net',
                                   dataset=self.dataset, dataloader=test_dataloader,
                                   save_predict_result_path=save_predict_result_path)
            result = inferncer.inference(self.inference_ckp)

        # 五折
        elif self.mode == "cv":
            collate_fn=None
            history = collections.defaultdict(list) 
            if self.dataset == 'fakesv':
                    data_split_dir = './data/FakeSV/data-split/event'
                    save_predict_result_path = './predict_result/FakeSV/'
            for fold in range(1, 6): 
                print('-' * 50)
                print ('fold %d:' % fold)
                print('-' * 50)

                train_data_path = data_split_dir + 'vid_fold_no_{i}.txt'.format(i = fold)
                test_data_path = data_split_dir + 'vid_fold_{i}.txt'.format(i = fold)

                data_load_time_start = time.time()
                self.model = SimpleMultiView(self.config)

                train_dataloader = self.get_dataloader(train_data_path)
                test_dataloader = self.get_dataloader(test_data_path)

                dataloaders = dict(zip(['train', 'test'], [train_dataloader, test_dataloader]))
                print('data load time: %.2f' % (time.time() - data_load_time_start))
                trainer = Trainer(model=self.model, device=self.device, lr=self.lr,
                              dataloaders=dataloaders,
                              epoches=self.epoches, model_name='MSAF_Net',
                              save_predict_result_path=save_predict_result_path,
                              loss_weight=self.loss_weight, early_stop=self.early_stop,
                              save_param_path=self.path_ckp + self.dataset + "/",
                              writer=SummaryWriter(self.path_tb + self.dataset + "/"))

                result = trainer.train()

                history['auc'].append(result['auc'])
                history['f1'].append(result['f1'])
                history['recall'].append(result['recall'])
                history['precision'].append(result['precision'])
                history['acc'].append(result['acc'])
                
            print ('results on 5-fold cross-validation: ')
            for metric in ['acc', 'f1', 'precision', 'recall', 'auc']:
                print ('%s : %.4f +/- %.4f' % (metric, np.mean(history[metric]), np.std(history[metric])))