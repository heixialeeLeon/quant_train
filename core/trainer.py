import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import time
import os
import numpy as np
from tqdm import tqdm
from core.utils.average import averager

class Trainner(object):
    def __init__(self, cfg, train_loader, val_loader, criterion, model, optimizer, scheduler):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.model.to(cfg.devices)
        self.devices = self.cfg.devices
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self):
        if self.cfg.resume != '':
            self._resume_model(self.cfg.resume)

        for epoch in range(self.cfg.epoch):
            self._train_epoch(epoch)
            if (epoch + 1) % self.cfg.save_interal == 0:
                model_path = self._create_epoch_model_name(epoch)
                self._save_model(model_path)
            self._eval()

    def _train_epoch(self, epoch):
        self.model.train()
        avg_loss = averager()
        sum_acc = 0
        sum_acc_count = 0
        pbar = tqdm(total=len(self.train_loader))
        for i, (data, label_p, label_c) in enumerate(self.train_loader):
            data, label_p, label_c = data.to(self.devices), label_p.to(self.devices), label_c.to(self.devices)
            label_c = torch.squeeze(label_c)
            self.optimizer.zero_grad()
            output = self.model.forward(data)
            minibatch, _ = output[0].size()
            loss_1 = self.criterion[0](output[0], label_p)
            loss_2 = self.criterion[1](output[1], label_c)
            loss = 10*loss_1 + loss_2
            avg_loss.add(loss)
            loss.backward()
            self.optimizer.step()

            p, val = output[1].max(1)
            acc = torch.sum(val == label_c).item()
            sum_acc += acc
            sum_acc_count += minibatch
            acc = sum_acc / sum_acc_count

            if i % self.cfg.display_interval == 0:
                show_msg = "epoch: {}, lr={:.6f} ,loss={:.4f}, class_acc={:.4f}".format(epoch, self._get_lr(), avg_loss.val(), acc)
                pbar.set_description(show_msg)
            pbar.update(1)
        self.scheduler.step()

    def _eval(self):
        self.model.eval()
        avg_loss = averager()
        sum_acc = 0
        sum_acc_count = 0
        for i, (data, label_p, label_c) in enumerate(self.val_loader):
            with torch.no_grad():
                data, label_p, label_c = data.to(self.devices), label_p.to(self.devices), label_c.to(self.devices)
                label_c = torch.squeeze(label_c)
                output = self.model.forward(data)
                minibatch, _ = output[0].size()
                loss_1 = self.criterion[0](output[0], label_p)
                loss_2 = self.criterion[1](output[1], label_c)
                loss = loss_1 + loss_2
                avg_loss.add(loss)

                p, val = output[1].max(1)
                acc = torch.sum(val == label_c).item()
                sum_acc += acc
                sum_acc_count += minibatch
                acc = sum_acc / sum_acc_count
        print("eval dataset loss={:.4f}, class_acc={:.4f}".format(avg_loss.val(), acc))

    def _create_epoch_model_name(self, epoch):
        if not os.path.exists(self.cfg.output):
            os.mkdir(self.cfg.output)
        return "{}/epoch_{}.pth".format(self.cfg.output, epoch)


    def _save_model(self, model_path):
        print('Saving checkpoint to: {}'.format(model_path))
        torch.save(self.model.state_dict(), model_path)

    def _resume_model(self, model_path):
        print("Resume model from {}".format(model_path))
        self.model.load_state_dict(torch.load(model_path))

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']