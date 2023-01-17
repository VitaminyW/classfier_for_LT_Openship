from Classfier import Classfier
from ShipDataset import ShipDataset
from trainer import Trainer
from torch.utils.data import DataLoader
from ClassBalanceSampler import ClassBalanceSampler
from until import *
import torch
from pathlib import Path
import argparse
import os

def train(opt):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = ShipDataset(opt.data_root_path,opt.train_file)
    val_dataset = ShipDataset(opt.data_root_path,opt.val_file,mode='val')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=False,
                          sampler=ClassBalanceSampler(train_dataset, num_samples_cls=1),
                          num_workers=opt.num_workers)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    model = Classfier()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=opt.lr) # 优化器管理模型的参数
    for para in list(model.backbone.parameters())[:-2]: # W b
        para.requires_grad = False
    if opt.n_GPUs > 1:#多GPU并行
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(opt.n_GPUs)])
    trainer = Trainer(model,optim,train_dataloader,val_dataloader,opt.times_per_eval,device)
    trainer(opt.epochs)
    
def test(opt):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_dataset = ShipDataset(opt.data_root_path,opt.test_file,mode='val')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    model = Classfier()
    model.to(device)
    weights_dict = torch.load(opt.checkpoint_path, map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                        if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(load_weights_dict, strict=False)
    preds = []
    indexs = []
    model.eval()
    for batch_index, (sample, label, index) in enumerate(test_dataloader):
        sample, label = sample.to(device), label.to(device)
        yp = model(sample)
        pred = yp.argmax(dim=1)
        preds.append(pred.detach().cpu().item())
        indexs.append(index.detach().cpu().item())
    # lazy loading module
    import pandas as pd
    preds_list = preds
    dataframe = [['img_id','label']]
    temp = []
    for index,item in enumerate(preds_list):
        filename = Path(test_dataset.img_path[indexs[index]]).stem
        temp.append([f'{filename}.tif',str(int(item))])
    temp.sort(key=lambda item:int(item[0].split('.')[0]))
    dataframe.extend(temp)
    dataframe = pd.DataFrame(dataframe)
    dataframe.to_csv('./test.csv', index=False, header=0)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', type=str, default='/home/yewei/openSARship')
    parser.add_argument('--train_file', type=str,
                        default='/home/yewei/classifier-balancing-main/data/OpenSARship_LT/OpenSARship_LT_train.txt')
    parser.add_argument('--val_file', type=str,
                        default='/home/yewei/classifier-balancing-main/data/OpenSARship_LT/OpenSARship_LT_val.txt')
    parser.add_argument('--test_file', type=str,
                        default='/home/yewei/classifier-balancing-main/data/OpenSARship_LT/OpenSARship_LT_test.txt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--n_GPUs', type=int, default=4)
    parser.add_argument('--times_per_eval', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--run_test', type=bool, default=True)
    parser.add_argument('--checkpoint_path', type=str, default='./params/model-95.pth')
    opt = parser.parse_args()
    if opt.run_test:
        test(opt)
    else:
        train(opt)