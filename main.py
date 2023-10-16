import argparse
import logging
import random
import warnings
import torch
import numpy as np

import utils
import data_loader as data_loader
from train_kd import train_and_evaluate, train_and_evaluate_kd
from model import resnet, mobilenetv2, shufflenetv2

random.seed(230)
torch.manual_seed(230)
np.random.seed(230)
torch.cuda.manual_seed(230)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default=None,
                    help="Directory containing params.json")
parser.add_argument('--num_class', default=100,
                    type=int, help="Number of classes")
parser.add_argument('-warm', type=int, default=1,
                    help='Warm up training phase')


def main():
    args = parser.parse_args()
    params = utils.args_to_dict(args.model_dir)

    utils.set_logger(args.model_dir)
    logging.info("Loading the datasets...")

    train_dl = data_loader.fetch_dataloader('train', params)
    dev_dl = data_loader.fetch_dataloader('dev', params)

    teacher_mapping = {
        "mobilenet_v2": mobilenetv2.mobilenetv2,
        "shufflenet_v2": shufflenetv2.shufflenetv2,
        "resnet18": resnet.ResNet18,
        "resnet50": resnet.ResNet50,
        "resnet101": resnet.ResNet101,
        "resnet152": resnet.ResNet152
    }

    if "distill" in params['model_version']:
        logging.info("KD Training...")

        student_mapping = {
            "shufflenet_v2_distill": shufflenetv2.shufflenetv2,
            "mobilenet_v2_distill": mobilenetv2.mobilenetv2,
            "resnet18_distill": resnet.ResNet18,
            "resnet50_distill": resnet.ResNet50
        }

        print(f'Student model: {params["model_version"]}')
        model = student_mapping.get(params['model_version'])(
            num_classes=args.num_class).cuda()

        print(f'Teacher model: {params["teacher"]}')
        teacher_model = teacher_mapping.get(params['teacher'])(
            num_classes=args.num_class).cuda()
        teacher_checkpoint = f'experiments/pretrained_teacher_models/base_{params["teacher"]}/best.pth.tar'

        optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'] * (params['batch_size'] / 128), momentum=0.9,
                                    weight_decay=5e-4)

        iter_per_epoch = len(train_dl)
        warmup_scheduler = utils.WarmUpLR(
            optimizer, iter_per_epoch * args.warm)

        utils.load_checkpoint(teacher_checkpoint, teacher_model)

        logging.info(f'Starting training for {params["num_epochs"]} epoch(s)')
        train_and_evaluate_kd(model, teacher_model, train_dl, dev_dl, optimizer, utils.loss_kd,
                              warmup_scheduler, params, args)

    else:
        logging.info("Normal Training...")

        print(f'Model: {params["model_version"]}')
        model = teacher_mapping.get(params['model_version'])(
            num_classes=args.num_class).cuda()

        optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'] * (params['batch_size'] / 128), momentum=0.9,
                                    weight_decay=5e-4)

        iter_per_epoch = len(train_dl)
        warmup_scheduler = utils.WarmUpLR(
            optimizer, iter_per_epoch * args.warm)

        logging.info(f'Starting training for {params["num_epochs"]} epoch(s)')
        train_and_evaluate(model, train_dl, dev_dl, optimizer, torch.nn.CrossEntropyLoss(), params,
                           args.model_dir, warmup_scheduler, args)


if __name__ == '__main__':
    main()
