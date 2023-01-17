from until import *
import tqdm
import torch
import os 
import sys

class Trainer:
    def __init__(self,model,optim,train_dataloader,val_dataloader,times_per_eval,device):
        self.model = model
        self.optim = optim
        self.logger = Logger()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.times_per_eval = times_per_eval
        self.device = device
    
    def train_one_epoch(self,epoch):
        self.model.train()
        loader_train_tqdm = tqdm.tqdm(self.train_dataloader)
        device = self.device
        avg_loss = torch.zeros(1).to(device)
        for batch_index, (sample, label, index) in enumerate(loader_train_tqdm):
            sample = sample.to(device)
            label = label.to(device)
            yp = self.model(sample)
            loss = self.loss_func(yp, label)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            avg_loss = (avg_loss * batch_index + loss.detach().cpu().item()) / (batch_index + 1)
            loader_train_tqdm.desc = f'[{epoch}] [{batch_index}/{len(self.train_dataloader)}]avg_loss = {avg_loss.item()}'
    
    @torch.no_grad()
    def eval_model(self,epoch):
        self.model.eval()
        device = self.device
        total_labels = torch.empty(0, dtype=torch.long).to(device)
        total_preds = torch.empty(0, dtype=torch.long).to(device)
        loader_val_tqdm = tqdm.tqdm(self.val_dataloader)
        for batch_index, (sample, label, index) in enumerate(loader_val_tqdm):
            sample, label = sample.to(device), label.to(device)
            yp = self.model(sample)
            preds = yp.argmax(dim=1)
            total_labels = torch.cat((total_labels, label))
            total_preds = torch.cat((total_preds, preds))
        eval_acc_mic_top1 = mic_acc_cal(total_preds, total_labels)
        eval_f_measure = F_measure(total_preds, total_labels)
        cls_accs = shot_acc(total_preds, total_labels)
        self.logger(f'eval[{epoch}]:' + (','.join([f'class {index}:{acc}' for index, acc in enumerate(cls_accs)])))
        self.logger(f'eval_acc_mic_top1:{eval_acc_mic_top1}')
        self.logger(f'eval_f_measure:{eval_f_measure}')
        print(f'eval[{epoch}]:' + (','.join([f'class {index}:{acc}' for index, acc in enumerate(cls_accs)])),
            file=sys.stdout)
        print(f'eval_acc_mic_top1:{eval_acc_mic_top1}', file=sys.stdout)
        print(f'eval_f_measure:{eval_f_measure}', file=sys.stdout)
        
    def __call__(self,num_epoch):
        for epoch in range(num_epoch):
            self.train_one_epoch(epoch)
            if (epoch + 1) % self.times_per_eval == 0:
                self.eval_model(epoch)
            if not os.path.exists('./params/'):
                os.mkdir('./params/')
            torch.save(self.model.module.state_dict(),
                    "./params/model-{epoch}.pth".format(epoch=epoch))