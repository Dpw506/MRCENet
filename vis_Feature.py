import argparse

import cv2
import numpy as np
import os
import torch
import time
from PIL import Image, ImageOps
#from models.vmamba_Fusion_efficross import VSSM_Fusion as net
from model.MRCENet import MRCENet as net
from Datasets.Dataset_load import Dataset_test
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import imageio
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from loss import Fusionloss
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='./data/')
    parser.add_argument('--output_folder', type=str, default='./output/')
    parser.add_argument('--model_path', type=str, default='./checkpoint/PET-MRI/Net_v27_11010_n3/fusion_model.pth')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mode', type=str, help='CT-MRI/PET-MRI/SPECT-MRI/Whu_Data_82', default='PET-MRI')
    parser.add_argument('--imagenet_model', type=str, default='imagenet')
    parser.add_argument('--gpu', type=str, help='0,1', default='0')
    return parser.parse_args()

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col

def visualize_feature_map(img_batch, out_path):
    feature_map = img_batch.cpu().clone().squeeze(0).permute(1,2,0)
    print(feature_map.shape)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        #plt.subplot(row, col, i + 1)
        #sns.heatmap(feature_map_split.detach())
        #plt.imshow(feature_map_split.detach())
        #plt.show()
        #plt.colorbar()
        # if i % 10 ==0:
        #plt.savefig('./results/feature/feature_map_{}.png'.format(i))
        #plt.axis('off')

    # plt.savefig(out_path + '/expert0_feature_map.png')
    # plt.show()

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination) / len(feature_map_combination)
    #sns.heatmap(feature_map_sum.detach().numpy(), cmap='jet')
    plt.imshow(feature_map_sum.detach().numpy())
    #plt.colorbar()
    plt.axis('off')
    plt.savefig(out_path + '/feature_map_cross_1.png')
    plt.show()

features = {}
def hook_fn(module, input, output):
    features['feat'] = output[1].detach().cpu()

def fusion(args):
    model = net()
    # tokenizer = BertTokenizer.from_pretrained("/home/evlab/dpw/Medical_image/Image_Fusion/FusionMamba-main/bert-base-uncased")
    # bert_model = BertModel.from_pretrained("/home/evlab/dpw/Medical_image/Image_Fusion/FusionMamba-main/bert-base-uncased")
    # bert_model = bert_model.cuda()
    model_path = args.model_path  # put the path of model.pth
    Method = 'Net_v27_11010_n3'
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model = model.to(device)
        model.load_state_dict(torch.load(model_path))
    else:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)

    output_features_folder = args.output_folder + args.mode + '/' + Method + '_Features/'
    os.makedirs(output_features_folder, exist_ok=True)

    test_mri = os.path.join(args.test_dir, args.mode, 'test/Feature', args.mode.split('-')[1])
    test_spect = os.path.join(args.test_dir, args.mode, 'test/Feature', args.mode.split('-')[0])

    # test_mri = os.path.join(args.test_dir, args.mode, 'test/Feature', 'MRI')
    # test_spect = os.path.join(args.test_dir, args.mode, 'test/Feature', 'PET')

    test_dataset = Dataset_test(test_mri, test_spect)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    tic = time.time()
    # criteria_fusion = Fusionloss()
    with torch.no_grad():
        model.eval()
        layer = model.cross_attention # 改成你想看的层
        hook = layer.register_forward_hook(hook_fn)
        for i, (image_ir, image_vis, filename) in enumerate(test_loader):
            img_1 = image_vis.to(device)
            img_2 = image_ir.to(device)
            fusion_image = model(img_1, img_2)
            #fusion_image = torch.clamp(fusion_image, 0, 1)

            # ones = torch.ones_like(fusion_image)
            # zeros = torch.zeros_like(fusion_image)
            # fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            # fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            visualize_feature_map(F.interpolate(input=features['feat'], size=img_1.shape[2:], mode='bilinear', align_corners=None),
                                  output_features_folder)

            fusion_out = fusion_image.detach().cpu().squeeze().numpy()
            #fusion_out = (fusion_out - np.min(fusion_out)) / (np.max(fusion_out) - np.min(fusion_out))

            result = (fusion_out * 255).astype(np.uint8)

            output_filename = filename[0]
            output_path = os.path.join(output_features_folder, output_filename)
            #imageio.imwrite(output_path, result)
            cv2.imwrite(output_path, result)


    toc = time.time()
    print('Processing time: {}'.format(toc - tic))

if __name__ == '__main__':
    args = parse_args()
    fusion(args)
