import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from utils.metrics import *
import copy
import pytorch_warmup as warmup
import pickle


class Trainer():
    def __init__(self, model, device, lr, dataloaders, save_param_path, writer, early_stop, epoches, model_name,
                 save_predict_result_path, loss_weight, scheduler_option=False, save_threshold=0.7, start_epoch=0):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.early_stop = early_stop
        self.save_threshold = save_threshold
        self.writer = writer
        self.scheduler_option = scheduler_option
        self.loss_weight = loss_weight

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path = save_param_path

        if os.path.exists(save_predict_result_path):
            self.save_predict_result_path = save_predict_result_path
        else:
            self.save_predict_result_path = os.makedirs(save_predict_result_path)
            self.save_predict_result_path = save_predict_result_path

        self.lr = lr

        # 定义 class_weights
        class_weights = torch.tensor([self.loss_weight, 1.0]).cuda()  # 例如类别权重
        # 使用加权损失函数
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.CEloss = nn.CrossEntropyLoss()
        # 调整学习率
        param_groups = [
            {"params": [p for p in self.model.parameters() if p is not self.model.w_2 and p is not self.model.w_3 and p is not self.model.b_1 and p is not self.model.b_2], "lr": self.lr},      # 模型的其他参数
            {"params": [self.model.w_2, self.model.w_3, self.model.b_1, self.model.b_2], "lr": self.lr},
        ]
        # 初始化优化器
        self.optimizer = torch.optim.Adam(param_groups, weight_decay=5e-5)
        if scheduler_option:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=1,
                min_lr=1e-6,
                verbose=True)

    def train(self):                        
        since = time.time()
        self.model.cuda()

        best_model_wts_test = copy.deepcopy(self.model.state_dict())
        best_f1_test = 0.0
        best_epoch_test = 0
        best_all_metrics = {}
        is_earlystop = False

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if is_earlystop:
                return best_all_metrics
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch, self.start_epoch + self.num_epochs - 1))
            print('-' * 50)

            # train phase
            self.model.train()
            print('-' * 10)
            print('TRAIN')
            print('-' * 10)
            running_loss = 0.0
            tpred = []
            tpred_before = []
            tlabel = []

            for batch in tqdm(self.dataloaders['train']):
                self.optimizer.zero_grad()
                batch_data = batch
                for k, v in batch_data.items():
                    if k != 'vid':
                        batch_data[k] = v.cuda()
                labels = batch_data['label']
                mix_output, _ = self.model(**batch_data)
                loss_CE = self.criterion(mix_output, labels)
                loss = loss_CE
                loss.backward()
                self.optimizer.step()
                print('Train Loss: {:.4f} '.format(loss_CE.item()))
                running_loss += loss.item() * labels.size(0)
                
                tpred.extend(torch.max(mix_output, 1)[1].tolist())
                tlabel.extend(labels.tolist())
                # with self.warmup_scheduler.dampening():
                #     self.scheduler.step()

            epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
            print('Train Loss: {:.4f} '.format(epoch_loss))
            results = metrics(tlabel, tpred)
            print(results)

            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            self.writer.add_scalar('Acc/train', results['acc'], epoch)
            self.writer.add_scalar('F1/train', results['f1'], epoch)

            # test phase
            self.model.eval()
            label = []
            vid = []
            pred = []

            final_fea = []

            for batch in tqdm(self.dataloaders['test']):
                batch_data = batch
                for k, v in batch_data.items():
                    if k != 'vid':
                        batch_data[k] = v.cuda()
                labels = batch_data['label']
                with torch.no_grad():
                    mix_output, batch_fea = self.model(**batch_data)
                    label.extend(labels.tolist())
                    pred.extend(torch.max(mix_output, 1)[1].tolist())
                    vid.extend(batch_data['vid'])
                    final_fea.extend(batch_fea.tolist())

            time_elapsed = time.time() - since
            print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print(get_confusionmatrix_fnd(np.array(pred), np.array(label)))
            test_results_val = metrics(label, pred)
            print("test pred result:", test_results_val)
            if test_results_val['f1'] > best_f1_test:
                best_f1_test = test_results_val['f1']
                best_all_metrics = test_results_val
                best_epoch_test = epoch
                best_model_wts_test = copy.deepcopy(self.model.state_dict())
                if best_f1_test > self.save_threshold:
                    final_save = {'vid': vid, 'final_fea': final_fea}
                    # 保存字典到 .pkl 文件
                    with open(self.save_predict_result_path + self.model_name + '.pkl', 'wb') as f:
                        pickle.dump(final_save, f)

                    torch.save(best_model_wts_test, self.save_param_path + self.model_name + "_test_" + str(
                        best_epoch_test) + "_{0:.4f}".format(test_results_val['f1']))
                    
                    results = []
                    for idx, _ in enumerate(pred):
                        record = {'vid': vid[idx], 'y_GT': label[idx], 'y_pred': pred[idx]}
                        results.append(record)
                    df = pd.DataFrame(results)
                    df.to_csv(self.save_predict_result_path + self.model_name + '.csv',index=False)

                    print("saved " + self.save_param_path + self.model_name + "_test_" + str(
                        best_epoch_test) + "_{0:.4f}".format(test_results_val['f1']))
            else:
                if epoch - best_epoch_test >= self.early_stop - 1:
                    is_earlystop = True
                    print("early stop at epoch " + str(epoch))
        return best_all_metrics

    def test(self, ckp_path):
        self.model.load_state_dict(torch.load(ckp_path))
        since = time.time()
        self.model.cuda()
        self.model.eval()
        pred = []
        label = []
        vid = []

        for batch in tqdm(self.dataloaders['test']):
            with torch.no_grad():
                batch_data = batch
                for k, v in batch_data.items():
                    if k != 'vid':
                        batch_data[k] = v.cuda()
                labels = batch_data['label']
                mix_output, text_only_feature, audio_only_feature, video_only_feature = self.model(**batch_data)
                label.extend(labels.tolist())
                pred.extend(torch.max(mix_output, 1)[1].tolist())

                vid.extend(batch_data['vid'])
        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        result = pd.DataFrame({'vid': vid, 'label': label, 'pred': pred})
        result.to_csv(self.save_predict_result_path + self.model_name + '.csv', sep='\t', index=False)

        print(get_confusionmatrix_fnd(np.array(pred), np.array(label)))
        print(metrics(label, pred))

        return metrics(label, pred)


class Inferencer():
    def __init__(self, model, device, model_name, dataset, dataloader, save_predict_result_path):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.dataset = dataset
        self.dataloader = dataloader
        if os.path.exists(save_predict_result_path):
            self.save_predict_result_path = save_predict_result_path
        else:
            self.save_predict_result_path = os.makedirs(save_predict_result_path)
            self.save_predict_result_path = save_predict_result_path

    def inference(self, ckp_path):
        self.model.load_state_dict(torch.load(ckp_path), strict=False)
        since = time.time()
        self.model.cuda()
        self.model.eval()

        label = []
        vid = []
        final_fea = []
        pred = []
        mix_f = []
        for batch in tqdm(self.dataloader):
            batch_data = batch
            for k, v in batch_data.items():
                if k != 'vid':
                    batch_data[k] = v.cuda()
            labels = batch_data['label']
            with torch.no_grad():
                mix_output, batch_fea = self.model(**batch_data)
                label.extend(labels.tolist())
                pred.extend(torch.max(mix_output, 1)[1].tolist())
                vid.extend(batch_data['vid'])
                final_fea.extend(batch_fea.tolist())
                mix_output = torch.softmax(mix_output, -1)
                mix_f.extend([mix_output[i][1].item() for i in range(len(mix_output))])

        final_save = {'vid': vid, 'final_fea': final_fea}
        # 保存字典到 .pkl 文件
        with open(self.save_predict_result_path + self.model_name + '.pkl', 'wb') as f:
            pickle.dump(final_save, f)

        time_elapsed = time.time() - since

        results = []
        for idx, _ in enumerate(pred):
            record = {'vid': vid[idx], 'y_GT': label[idx], 'y_pred': pred[idx]}
            results.append(record)
        df = pd.DataFrame(results)
        df.to_csv(self.save_predict_result_path + self.model_name + '.csv',index=False)

        print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print(get_confusionmatrix_fnd(np.array(pred), np.array(label)))
        print("test pred result:", metrics(label, pred))
        return metrics(label, pred)
