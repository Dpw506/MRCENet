import argparse

import cv2
import numpy as np
import os
import torch
import time
from PIL import Image, ImageOps

from model.MRCENet import MRCENet as net
from Datasets.Dataset_load import Dataset_test
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import imageio
# from loss import Fusionloss

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='./data/')
    parser.add_argument('--output_folder', type=str, default='./output/')
    parser.add_argument('--model_path', type=str, default='./checkpoint/SPECT-MRI/MRCENet_lgradint/fusion_model.pth')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mode', type=str, help='CT-MRI/PET-MRI/SPECT-MRI/Whu_Data_82', default='SPECT-MRI')
    parser.add_argument('--imagenet_model', type=str, default='imagenet')
    parser.add_argument('--gpu', type=str, help='0,1', default='0')
    return parser.parse_args()


def fusion(args):
    model = net()
    # tokenizer = BertTokenizer.from_pretrained("/home/evlab/dpw/Medical_image/Image_Fusion/FusionMamba-main/bert-base-uncased")
    # bert_model = BertModel.from_pretrained("/home/evlab/dpw/Medical_image/Image_Fusion/FusionMamba-main/bert-base-uncased")
    # bert_model = bert_model.cuda()
    model_path = args.model_path  # put the path of model.pth
    Method = 'MRCENet'
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model = model.to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)

    output_folder = args.output_folder + args.mode + '/' + Method + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if args.mode == "Whu_Data_82":
        test_mri = os.path.join(args.test_dir, args.mode, 'test/MRI' )
        test_spect = os.path.join(args.test_dir, args.mode, 'test/PET')
    else:
        test_mri = os.path.join(args.test_dir, args.mode, 'test', args.mode.split('-')[1])
        test_spect = os.path.join(args.test_dir, args.mode, 'test', args.mode.split('-')[0])

    # test_mri = '/home/evlab/dpw/Medical_image/Image_Fusion/MMIF_MY/data/Sample_11/MRI_11_nii2png'
    # test_spect = '/home/evlab/dpw/Medical_image/Image_Fusion/MMIF_MY/data/Sample_11/PET_11_nii2png'

    test_dataset = Dataset_test(test_mri, test_spect)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    tic = time.time()
    # criteria_fusion = Fusionloss()
    i = 0
    with torch.no_grad():
        model.eval()
        for i, (image_ir, image_vis, filename) in enumerate(test_loader):
            i += 1
            img_1 = image_vis.to(device)
            img_2 = image_ir.to(device)
            fusion_image = model(img_1, img_2)
            fusion_image = torch.clamp(fusion_image, 0, 1)

            # ones = torch.ones_like(fusion_image)
            # zeros = torch.zeros_like(fusion_image)
            # fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            # fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)

            fusion_out = fusion_image.detach().cpu().squeeze().numpy()
            #fusion_out = (fusion_out - np.min(fusion_out)) / (np.max(fusion_out) - np.min(fusion_out))

            result = (fusion_out * 255).astype(np.uint8)

            output_filename = filename[0]
            output_path = os.path.join(output_folder, output_filename)
            imageio.imwrite(output_path, result)
            #cv2.imwrite(output_path, result)


    toc = time.time()
    print('Processing time: {}'.format((toc - tic) / i))

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    fusion(args)
