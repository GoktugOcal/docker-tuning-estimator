import sys
import os
import json
import pickle
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.sampler as sampler
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from time import sleep, time

import logging

DEVICE = "cpu"

def load_data_cifar(dataset_path = "./data/", batch_size = 128):

    train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, 
        pin_memory=True,
        shuffle=True)#, sampler=train_sampler)


    val_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True,
    transform=transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))
    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True) #, sampler=valid_sampler)
    return train_loader, test_loader

def fine_tune(model, no_epoch, batch_size, train_loader, print_frequency=100):

    # Data loaders for fine tuning and evaluation.
    momentum = 0.9
    weight_decay = 1e-4
    finetune_lr = 0.001

    criterion = torch.nn.BCEWithLogitsLoss()
    
    _NUM_CLASSES = 10
    optimizer = torch.optim.SGD(
        model.parameters(),
        finetune_lr, 
        momentum=momentum,
        weight_decay=weight_decay)

    model = model.to(DEVICE)
    model.train()
    print("Fine tuning started.")
    durations = []
    for epoch_no in range(no_epoch):
        if epoch_no % print_frequency == 0:
            logging.info('Fine-tuning Epoch {}'.format(epoch_no))
            sys.stdout.flush()
        for i, (input, target) in enumerate(tqdm(train_loader)):

            start = time()
            
            # Ensure the target shape is sth like torch.Size([batch_size])
            if len(target.shape) > 1: target = target.reshape(len(target))

            target.unsqueeze_(1)
            target_onehot = torch.FloatTensor(target.shape[0], _NUM_CLASSES)
            target_onehot.zero_()
            target_onehot.scatter_(1, target, 1)
            target.squeeze_(1)
            input, target = input.to(DEVICE), target.to(DEVICE)
            target_onehot = target_onehot.to(DEVICE)

            pred = model(input)
            loss = criterion(pred, target_onehot)
            optimizer.zero_grad()
            loss.backward()  # compute gradient and do SGD step
            optimizer.step()

            del loss, pred

            duration = time() - start
            durations.append((epoch_no, duration))

            if(i == 10): break

    return model, durations


if __name__ == "__main__":


    arg_parser = ArgumentParser()
    arg_parser.add_argument('--cpus', type=int)
    arg_parser.add_argument('--batch_size', type=int)
    arg_parser.add_argument('--model_path', type=str)
    arg_parser.add_argument('--log_path', type=str)
    args = arg_parser.parse_args()

    debug_logfilename = "./logs/debug.txt"
    logging.basicConfig(
        filename=debug_logfilename,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info("Started.")
    

    model = torch.load(args.model_path)
    no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainloader, _ = load_data_cifar(batch_size=args.batch_size)

    _, durations = fine_tune(
        model,
        no_epoch=5,
        batch_size=args.batch_size,
        train_loader=trainloader,
        print_frequency=1)
    
    log_file = open(args.log_path, "a")
    
    for epoch_no, duration in durations:
        line = f"{args.cpus},{args.batch_size},{len(trainloader)},{no_params},{epoch_no},{duration}\n"
        log_file.write(line)
    
    logging.info("Done.")