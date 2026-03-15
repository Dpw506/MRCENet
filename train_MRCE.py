#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
from glob import glob
from model.MRCENet import MRCENet
import argparse
import datetime
import time
import logging

import os
from logger import setup_logger
from utils import to_ssim_skimage, MSE
from Datasets.Dataset_load import Dataset_train, Dataset_val

from tensorboardX import SummaryWriter

from loss import Fusionloss as Fusionloss
import random
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parse = argparse.ArgumentParser(description='Train with pytorch')
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('--batch_size', type=int, default=1)
    parse.add_argument('--lr', type=float, default=0.0001)
    parse.add_argument('--train_dir', type=str, default='./data/')
    parse.add_argument('--val_dir', type=str, default='./data/')
    parse.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parse.add_argument('--log_dir', type=str, default='./log')
    parse.add_argument('--imagenet_model', type=str, default='imagenet_model_dir')
    parse.add_argument('--mode', type=str, help='[CT-MRI/PET-MRI/SPECT-MRI/Whu_Data_82]', default='CT-MRI')
    parse.add_argument('--gpu', type=str, help='0,1', default='1')
    return parse.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_fusion(args, num=0, logger=None):
    best_ssim = 0
    best_mse = 10
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    epoch = args.epochs
    lr_start = args.lr
    modelpth = args.checkpoint_dir + '/' + args.mode
    Method = 'MRCENet'
    modelpth = os.path.join(modelpth, Method)
    os.makedirs(modelpth, exist_ok=True)
    fusionmodel = MRCENet()
    fusionmodel.to(device)
    train_mri = os.path.join(args.train_dir, args.mode, 'train', args.mode.split('-')[1])
    val_mri = os.path.join(args.val_dir, args.mode, 'val', args.mode.split('-')[1])
    train_spect = os.path.join(args.train_dir, args.mode, 'train', args.mode.split('-')[0])
    val_spect = os.path.join(args.val_dir, args.mode, 'val', args.mode.split('-')[0])

    optimizer = torch.optim.AdamW(fusionmodel.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-6)

    writer = SummaryWriter(log_dir=os.path.join(modelpth, 'tensorboard'))

    train_dataset = Dataset_train(ir_path=train_mri, vis_path=train_spect, data_augment=True)
    val_dataset = Dataset_val(ir_path=val_mri, vis_path=val_spect)
    print("the training dataset is length:{}".format(train_dataset.__len__()))
    print("the val dataset is length:{}".format(val_dataset.__len__()))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4)
    train_loader.n_iter = len(train_loader)
    criteria_fusion = Fusionloss()

    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    for epo in range(0, epoch):
        print('\n| epo #%s begin...' % epo)

        fusionmodel.train()
        for it, (image_ir, image_vis) in enumerate(train_loader):
            image_vis = image_vis.to(device)
            image_ir = image_ir.to(device)

            optimizer.zero_grad()
            fusion_image = fusionmodel(image_vis, image_ir)


            loss_fusion,  loss_in, ssim_loss, loss_grad = criteria_fusion(
                image_vis=image_vis, image_ir=image_ir, generate_img=
                fusion_image, weight0=0.5, weight1=0.5
            )

            loss_total = loss_fusion

            loss_total.backward()
            optimizer.step()
            ed = time.time()
            writer.add_scalars('Loss/fusion', {'training total loss': loss_total.item()
                                              }, train_loader.n_iter * epo + it + 1)
            writer.add_scalars('Loss/sub_loss', {'int_loss': loss_in.item(),
                                                               'ssim_loss': ssim_loss.item(),
                                                               'grad_loss': loss_grad.item(),
                                                               #'mi_loss': loss_m.item(),
                                                      }, train_loader.n_iter * epo + it + 1)
            writer.add_scalars('Learning_rate', {'learning_rate': optimizer.param_groups[0]['lr']},
                                                 train_loader.n_iter * epo + it + 1)

            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'ssim_loss: {loss_ssim:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_ssim=ssim_loss.item(),
                    #loss_m=loss_m.item(),
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)
                st = ed
        scheduler.step()
        if epo % 5 == 0:
            with torch.no_grad():
                fusionmodel.eval()
                ssim_list = []
                for it, (image_ir, image_vis) in enumerate(val_loader):
                    img_vis = image_vis.to(device)
                    img_ir = image_ir.to(device)
                    fusion_image = fusionmodel(img_vis, img_ir)

                    ssim_list.extend(to_ssim_skimage(img_ir, img_vis, fusion_image))

            avr_ssim = sum(ssim_list) / len(ssim_list)

            image_debug = torch.cat((fusion_image.detach().cpu(), image_ir.detach().cpu(), image_vis.detach().cpu()), dim=0)

            writer.add_images('fusion_image_vis', image_debug, epo)
            writer.add_scalars('val', {'val ssim': avr_ssim,
                                                            }, epo)

            if best_ssim < avr_ssim:
                best_ssim = max(avr_ssim, best_ssim)
                # args.best_psnr = max(avr_psnr, args.best_psnr)
                best_epoch = epo

                torch.save(fusionmodel.state_dict(), os.path.join(modelpth, 'fusion_model_best.pth'))
            logger.info(f'current epoch: {epo}\t'
                        f'current ssim: {avr_ssim:.4f}\t'
                        f'best ssim: {best_ssim:.4f}\t'
                        f'best epoch: {best_epoch}\t')

    fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
    #os.makedirs(modelpth, exist_ok=True)
    torch.save(fusionmodel.state_dict(), fusion_model_file)
    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')
    writer.close()

def main():
    args = parse_args()


    # 创建日志目录
    logpath = args.log_dir
    os.makedirs(logpath, exist_ok=True)

    # 配置 logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    setup_logger(logpath)

    logger.info(f"Arguments: {args}")
    setup_seed(1337)

    # 训练循环（当前仅 1 次）
    for i in range(1):
        train_fusion(args, i, logger)
        logger.info(f"Train Fusion Model {i + 1} Successfully!")

    logger.info("Training Done!")



if __name__ == "__main__":
    main()
