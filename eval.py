# -*- coding: UTF-8 -*-
'''
@author: mengting gu
@contact: 1065504814@qq.com
@time: 2021/2/19 16:57
@file: eval.py
@desc: 
'''
# -*-coding:utf-8-*-
import argparse
import logging

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import yaml
from easydict import EasyDict

from models import get_model
from utils import (
    Logger,
    count_parameters,
    data_augmentation,
    get_data_loader,
)
from data import image_processing
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch CIFAR Dataset Training")
parser.add_argument("--work-path", default= "./experiments/cifar10/vgg19", type=str)
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")

args = parser.parse_args()
logger = Logger(
    log_file_name=args.work_path + "/log.txt",
    log_level=logging.DEBUG,
    logger_name="CIFAR",
).get_log()
config = None


def eval(test_loader, net, device):

    net.eval()

    correct = 0
    total = 0

    logger.info(" === Validate ===")

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze()
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # logger.info(
            #     "   == test acc: {:6.3f}% | true label : {}, predict as : {}".format(
            #          100.0 * correct / total, targets, predicted
            #     )
            # )
        logger.info(
            "   == test acc: {:6.3f}% , model best prec : {:6.3f}%".format(
                100.0 * correct / total, best_prec
            )
        )

def one_image_demo(image_path, net, device):
    net.eval()
    img = image_processing.read_image(image_path, resize_height=config.input_size, resize_width=config.input_size)
    img = transforms.ToTensor()(img)
    img = img[np.newaxis, :, :, :]
    inputs = img.to(device)
    outputs = net(inputs)
    _, predicted = outputs.max(1)
    print("img : {}, predict as : {}".format(image_path, predicted[0]))


def main():
    global args, config, best_prec

    # read config from yaml file
    with open(args.work_path + "/config.yaml") as f:
        config = yaml.load(f)
    # convert to dict
    config = EasyDict(config)
    logger.info(config)

    # define netowrk
    net = get_model(config)
    ckpt_file_name = args.work_path + "/" + config.ckpt_name + "_best.pth.tar"
    checkpoint = torch.load(ckpt_file_name)
    net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint["state_dict"].items()}, strict=True)
    best_prec = checkpoint["best_prec"]
    logger.info(net)
    logger.info(" == total parameters: " + str(count_parameters(net)))

    # CPU or GPU
    device = "cuda" if config.use_gpu else "cpu"
    # data parallel for multiple-GPU
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.to(device)

    # load training data, do data augmentation and get data loader
    transform_train = transforms.Compose(data_augmentation(config))
    transform_test = transforms.Compose(data_augmentation(config, is_train=False))
    train_loader, test_loader = get_data_loader(transform_train, transform_test, config)
    eval(test_loader, net, device)


    # one_image_demo demo
    # image_path = "./data/cifar.jpg"
    # one_image_demo(image_path, net, device)


if __name__ == "__main__":
    main()