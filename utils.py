import json
import logging
import os
import shutil
import torch
from torch.optim.lr_scheduler import _LRScheduler

import torch.nn as nn
import torch.nn.functional as F


def loss_kd(outputs, labels, teacher_outputs, params):
    kl_div = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/params.temperature, dim=1),
                                                 F.softmax(teacher_outputs/params.temperature, dim=1)) * (params.temperature ** 2)
    return (1. - params.alpha) * F.cross_entropy(outputs, labels) + params.alpha * kl_div


class Params():
    def __init__(self, json_path):
        with open(json_path) as f:
            self.__dict__.update(json.load(f))

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            self.__dict__.update(json.load(f))

    @property
    def dict(self):
        return self.__dict__


class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint, epoch_checkpoint=False):
    filepath = os.path.join(checkpoint, 'last.pth.tar')

    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")

    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
    if epoch_checkpoint == True:
        epoch_file = str(state['epoch'] - 1) + '.pth.tar'
        shutil.copyfile(filepath, os.path.join(checkpoint, epoch_file))


def load_checkpoint(checkpoint, model, optimizer=None):
    if not os.path.exists(checkpoint):
        raise Exception("File doesn't exist {}".format(checkpoint))

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(
            checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
