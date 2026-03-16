import os
from torch.utils.data import Dataset
import numpy as np
import PIL.Image as Image
import random
from torchvision import transforms
import shutil
import torchvision.transforms.functional as TF
import pywt
from math import exp

def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for dir in dirnames:
            files = os.listdir(os.path.join(parent, dir))
            for file in files:
                if file.lower().endswith(
                        ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                    imagelist.append(os.path.join(parent, dir, file))
        return imagelist

def augment(ir, vis):
    augmentation_method = random.choice([0, 1, 2, 3, 4, 5])
    #augmentation_method = random.choice([0])
    rotate_degree = random.choice([90, 180, 270])
    '''Rotate'''
    if augmentation_method == 0:
        hazy = transforms.functional.rotate(ir, rotate_degree)
        clean = transforms.functional.rotate(vis, rotate_degree)
        return hazy, clean
    '''Vertical'''
    if augmentation_method == 1:
        vertical_flip = transforms.RandomVerticalFlip(p=1)
        hazy = vertical_flip(ir)
        clean = vertical_flip(vis)
        return hazy, clean
    '''Horizontal'''
    if augmentation_method == 2:
        horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        hazy = horizontal_flip(ir)
        clean = horizontal_flip(vis)
        return hazy, clean
    '''no change'''
    if augmentation_method == 3 or augmentation_method == 4 or augmentation_method == 5:
        return ir, vis
    
def entropy(X):
    X = X.flatten()
    X = np.uint8(X)
    n = len(X)
    counts = np.bincount(X)
    probs = counts[np.nonzero(counts)]/n
    en = 0
    for i in range(len(probs)):
        en = en - probs[i] * np.log(probs[i]/np.log(2))
    return en

def haar_weight(MRI, OTHER):
    cA, (cH, cV, cD) = pywt.dwt2(MRI, 'haar')
    MRI_SUM = (entropy(cH) + entropy(cV) + entropy(cD)) / 3.0
    cA, (cH, cV, cD) = pywt.dwt2(OTHER, 'haar')
    # OTHERI_SUM = abs(np.average(cH)) + abs(np.average(cV)) + abs(np.average(cD))
    # OTHERI_SUM = (cH.std() + cV.std() + cD.std()) / 3
    OTHERI_SUM = (entropy(cH) + entropy(cV) + entropy(cD))/3.0

    MRI_SUM, OTHERI_SUM = MRI_SUM/(OTHERI_SUM+MRI_SUM)/3, OTHERI_SUM/(OTHERI_SUM+MRI_SUM)/3

    MRI_weight = exp(MRI_SUM) / (exp(MRI_SUM) + exp(OTHERI_SUM))
    OTHERI_weight = exp(OTHERI_SUM) / (exp(MRI_SUM) + exp(OTHERI_SUM))

    return MRI_weight, OTHERI_weight

class Dataset_train(Dataset):
    def __init__(self, ir_path, vis_path, data_augment=False):
        self.ir_path = ir_path
        self.vis_path = vis_path
        self.data_augment = data_augment
        self.imagelist = os.listdir(ir_path)

        self.basic_transform = transforms.ToTensor()
        self.augment_transform = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(),  # 水平翻转
                transforms.RandomVerticalFlip(),  # 垂直翻转
                transforms.RandomRotation(0),  # 轻微旋转
                transforms.RandomRotation(90),  # 轻微旋转 
                transforms.RandomRotation(180),
                transforms.RandomRotation(270),
            ]),
            transforms.ToTensor()])
        

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, index):
        ir_path = os.path.join(self.ir_path, self.imagelist[index])
        vis_path = os.path.join(self.vis_path, self.imagelist[index])
        ir_img = Image.open(ir_path).convert('L')
        vis_img = Image.open(vis_path).convert('L')

        i, j, h, w = transforms.RandomCrop.get_params(ir_img, output_size=(128, 128))
        ir_img = TF.crop(ir_img, i, j, h, w)
        vis_img = TF.crop(vis_img, i, j, h, w)

        #mri_weight, other_weight = haar_weight(np.array(ir_img), np.array(vis_img))

        if self.data_augment:
            ir_img = self.augment_transform(ir_img)
            vis_img = self.augment_transform(vis_img)
        else:
            ir_img = self.basic_transform(ir_img)
            vis_img = self.basic_transform(vis_img)

        return ir_img, vis_img#, mri_weight, other_weight

class Dataset_val(Dataset):
    def __init__(self, ir_path, vis_path):
        self.ir_path = ir_path
        self.vis_path = vis_path

        self.imagelist = os.listdir(ir_path)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, index):
        ir_path = os.path.join(self.ir_path, self.imagelist[index])
        vis_path = os.path.join(self.vis_path, self.imagelist[index])
        ir_img = Image.open(ir_path).convert('L')
        vis_img = Image.open(vis_path).convert('L')
        ir = self.transform(ir_img)
        vis = self.transform(vis_img)
        return ir, vis


class Dataset_test(Dataset):
    def __init__(self, ir_path, vis_path):
        self.ir_path = ir_path
        self.vis_path = vis_path
        self.imagelist = os.listdir(ir_path)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, index):
        ir_path = os.path.join(self.ir_path, self.imagelist[index])
        vis_path = os.path.join(self.vis_path, self.imagelist[index])
        filename = self.imagelist[index]

        ir_img = Image.open(ir_path).convert('L')
        vis_img = Image.open(vis_path).convert('L')
        ir = self.transform(ir_img)
        vis = self.transform(vis_img)
        return ir, vis, filename


if __name__ == '__main__':
    CT_PATH = '/home/dpw/Dpw/code/Medical_Images/Image_Fusion/MMIF_MY/data/CT_MRI_Original/CT/'
    MRI_PATH = '/home/dpw/Dpw/code/Medical_Images/Image_Fusion/MMIF_MY/data/CT_MRI_Original/MRI'

    PETlist = get_img_file(CT_PATH)
    MRIlist = get_img_file(MRI_PATH)

    assert len(PETlist) == len(MRIlist), (f'The lengths of the {CT_PATH} and {MRI_PATH} datasets are inconsistent. '
                                                        f'Please make sure that the datasets are correct.')

    PET_train = '/home/dpw/Dpw/code/Medical_Images/Image_Fusion/MMIF_MY/data/CT-MRI/train/CT'
    os.makedirs(PET_train, exist_ok=True)
    MRI_train = '/home/dpw/Dpw/code/Medical_Images/Image_Fusion/MMIF_MY/data/CT-MRI/train/MRI'
    os.makedirs(MRI_train, exist_ok=True)

    PET_val = '/home/dpw/Dpw/code/Medical_Images/Image_Fusion/MMIF_MY/data/CT-MRI/val/CT'
    os.makedirs(PET_val, exist_ok=True)
    MRI_val = '/home/dpw/Dpw/code/Medical_Images/Image_Fusion/MMIF_MY/data/CT-MRI/val/MRI'
    os.makedirs(MRI_val, exist_ok=True)

    PET_test = '/home/dpw/Dpw/code/Medical_Images/Image_Fusion/MMIF_MY/data/CT-MRI/test/CT'
    os.makedirs(PET_test, exist_ok=True)
    MRI_test = '/home/dpw/Dpw/code/Medical_Images/Image_Fusion/MMIF_MY/data/CT-MRI/test/MRI'
    os.makedirs(MRI_test, exist_ok=True)

    l = len(PETlist)
    # for i, filepath in enumerate(PETlist):
    #     pet = Image.open(filepath).convert('RGB')
    #     print(pet.size)
    #     mri = Image.open(MRI_PATH + '/' + filepath.split('/')[-2].replace('PET', 'MRI') + '/' + filepath.split('/')[-1]).convert('RGB')
    #     print(mri.size)
    #     if pet.size != mri.size:
    #         pet.resize(mri.size, resample='bilinear')
    #     pet.save(PET_train + '/' + f'{i}.png')
    for i, (filepath, mripath) in enumerate(zip(PETlist, MRIlist)):
        # pet = Image.open(filepath).convert('RGB')
        # print(pet.size)
        # mri = Image.open(
        #     MRI_PATH + '/' + filepath.split('/')[-2].replace('SPECT', 'MRI') + '/' + filepath.split('/')[-1]).convert(
        #     'RGB')
        # print(mri.size)
        # if pet.size != mri.size:
        #     pet.resize(mri.size, resample=Image.Resampling.BILINEAR)
        if i < l * 0.7:
            #pet.save(PET_train + '/' + str(i) + '.png')
            shutil.copy(filepath, PET_train + '/' + f'{i}.png')
            shutil.copy(MRI_PATH + '/' + filepath.split('/')[-2].replace('CT', 'MRI') + '/' + filepath.split('/')[-1],
                        MRI_train + '/' + f'{i}.png')
        elif i > l * 0.7 and i < l * 0.8:
            #pet.save(PET_val + '/' + str(i) + '.png')
            shutil.copy(filepath, PET_val + '/' + f'{i}.png')
            shutil.copy(MRI_PATH + '/' + filepath.split('/')[-2].replace('CT', 'MRI') + '/' + filepath.split('/')[-1],
                        MRI_val + '/' + f'{i}.png')
        elif i > l * 0.8:
            #pet.save(PET_test + '/' + str(i) + '.png')
            shutil.copy(filepath, PET_test + '/' + f'{i}.png')
            shutil.copy(MRI_PATH + '/' + filepath.split('/')[-2].replace('CT', 'MRI') + '/' + filepath.split('/')[-1],
                        MRI_test + '/' + f'{i}.png')
    #print(file_path)