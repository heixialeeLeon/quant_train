import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import argparse
from torch.utils.data import DataLoader
#from core.data.datacsv import StockData
from core.data.dailyData import StockDailyData
from core.model.lstm import LSTM
from core.trainer import Trainner
from core.utils.config import get_config
import random
import numpy as np

def get_optimizer(cfg, model):
    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), cfg.lr, amsgrad=True)
    elif cfg.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
    return optimizer

def get_scheduler(optimzer, cfg):
    if cfg.name == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(optimzer, **cfg.args)
        return scheduler
    elif cfg.name == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimzer, **cfg.args)
        return scheduler
    else:
        raise KeyError("unknown scheduler type: {}".format(cfg.name))

def get_train_val_data(cfg):
    train_dataset = StockDailyData(cfg, train_flag=True)
    val_dataset = StockDailyData(cfg, train_flag=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    return train_dataloader, val_dataloader

def get_model(cfg):
    return LSTM(cfg.input_feature)

def train(config):
    # random setting
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(config.random_seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.random_seed)  # 为所有GPU设置随机种子

    train_dataloader, val_dataloader = get_train_val_data(config.data)
    model = get_model(config.model)
    optimizer = get_optimizer(config.train, model)
    criterion = [nn.L1Loss(), nn.CrossEntropyLoss()]
    scheduler = get_scheduler(optimizer, config.train.scheduler)
    trainer = Trainner(config.train,train_dataloader, val_dataloader, criterion, model, optimizer, scheduler)
    trainer.fit()

def main():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', default='configs/config.py',type=str, help='train config file path')
    args = parser.parse_args()
    cfg = get_config(args.config)
    print(cfg)
    train(cfg)

if __name__ == "__main__":
    main()
