import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import Params
from pathlib import Path
import csv
from datasets.augmentations import get_augmentations
import csv
from sklearn.metrics import f1_score,jaccard_score
from pathlib import Path
from catalyst import utils
from catalyst.dl import SupervisedRunner
from datasets.dataset import SegmentationDataset
import matplotlib.pyplot as plt

params=Params("./params.json")

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset import SegmentationDataset
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    images = sorted(Path(params.train_image_path).glob("*.png"))
    masks = sorted(Path(params.train_mask_path).glob("*.png"))
    train_path=params.train_id_path
    val_path=params.val_id_path
    test_path=params.test_id_path
    train_images,train_masks=get_split(train_path,images,masks)
    print("Generated trainingset...")
    val_images,val_masks = get_split(val_path,images,masks)
    print("Generated validationset...")
    test_images,test_masks=get_split(test_path,images,masks)

    transforms,_,_=get_augmentations(params)
    db_train = SegmentationDataset(train_images,
        num_classes=params.num_classes,
        image_size=params.img_shape, #only squares right now 
        masks=train_masks,
        transforms=transforms,
        testmode=False)
    db_val = SegmentationDataset(val_images,
        num_classes=params.num_classes,
        image_size=params.img_shape, #only squares right now 
        masks=val_masks,
        transforms=transforms,
        testmode=False)

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    best_val_loss=100
    train_dice_losses=[]
    train_ce_losses=[]
    val_dice_losses=[]
    val_ce_losses=[]

    for epoch_num in iterator:
        train_dice_loss_epoch=0
        train_ce_loss_epoch=0
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['mask_ce']
            if torch.cuda.is_available():
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            train_dice_loss_epoch+=0.5*loss_dice.item()
            train_ce_loss_epoch+=0.5*loss_ce.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            print("\r {}".format("Iteration " +str(iter_num) + ", loss: " + str(loss.item())+ ", loss_ce: " +str(loss_ce.item())), end="")
            #logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            # if iter_num % 20 == 0:
            #     image = image_batch[1, 0:1, :, :]
            #     image = (image - image.min()) / (image.max() - image.min())
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)
        train_ce_loss_epoch=train_ce_loss_epoch/len(db_train)
        train_dice_loss_epoch=train_dice_loss_epoch/len(db_train)

        train_dice_losses.append(train_dice_loss_epoch)
        train_ce_losses.append(train_ce_loss_epoch)
        total_loss=train_dice_loss_epoch+train_ce_loss_epoch

        val_dice_loss_epoch, val_ce_loss_epoch= get_validation_loss(model,db_val,batch_size)
        total_val_loss=val_dice_loss_epoch+ val_ce_loss_epoch
        val_dice_losses.append(val_dice_loss_epoch)
        val_ce_losses.append(val_ce_loss_epoch)

        plot_loss(train_dice_losses,val_dice_losses,"Dice Loss")
        plot_loss(train_ce_losses,val_ce_losses,"Cross Entropy Loss")
        plot_loss(np.array(train_dice_losses)+np.array(train_ce_losses),np.array(val_dice_losses)+np.array(val_ce_losses),"Total Loss")

        print("\n trainloss: ", total_loss)
        print("validationloss: ", total_val_loss)

        if(total_val_loss<best_val_loss):
            best_val_loss=total_val_loss
            print("New best validation loss.")
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        save_interval = int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
        #here:validation

    writer.close()
    return "Training Finished!"

def plot_loss(train_loss,val_loss, name):
    plt.plot(train_loss,label="training loss")
    plt.plot(val_loss, label="validation loss")
    plt.legend()
    plt.xticks(range(0,len(train_loss),5))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(name)
    plt.savefig(name+"_plot.png")
    plt.close()

def get_validation_loss(net,db_val,batch_size):
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(params.num_classes)
    testloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=1)
    total_loss_dice=0
    total_loss_ce=0
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(testloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['mask_ce']
            if torch.cuda.is_available():
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = net(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            total_loss_dice+=0.5*loss_dice.item()
            total_loss_ce+=0.5*loss_ce.item()

    return total_loss_dice/len(db_val),total_loss_ce/len(db_val)
    
def get_split(filepath,im_array,mask_array):
    new_mask_arr=[]
    new_im_arr=[]
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            for im in im_array:
                if(row[1] in str(im)):
                    new_im_arr.append(im)
                    break
            for mask in mask_array:
                if(row[1] in str(mask)):
                    new_mask_arr.append(mask)
                    break
    return new_im_arr,new_mask_arr