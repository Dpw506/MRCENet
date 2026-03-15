import numpy as np
import torch
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from Evaluator import Evaluator

def to_ssim_skimage(ir, vis, fi):
    ir_list = torch.split(ir, 1, dim=0)
    vis_list = torch.split(vis, 1, dim=0)
    fi_list = torch.split(fi, 1, dim=0)

    ir_list_np = [ir_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(ir_list))]
    vis_list_np = [vis_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(vis_list))]
    fi_list_np = [fi_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(fi_list))]
    ssim_list = [ssim(fi_list_np[ind], ir_list_np[ind], data_range=1) + ssim(fi_list_np[ind], vis_list_np[ind], data_range=1) for ind in range(len(fi_list))]

    return ssim_list

def to_SCD(ir, vis, fi):
    ir_list = torch.split(ir, 1, dim=0)
    vis_list = torch.split(vis, 1, dim=0)
    fi_list = torch.split(fi, 1, dim=0)

    ir_list_np = [ir_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(ir_list))]
    vis_list_np = [vis_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(vis_list))]
    fi_list_np = [fi_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(fi_list))]
    ssim_list = [Evaluator.SCD((fi_list_np[ind] * 255).astype(np.uint8), (ir_list_np[ind] * 255).astype(np.uint8), (vis_list_np[ind]* 255).astype(np.uint8)) for ind in range(len(fi_list))]

    return ssim_list

def MSE(fusion, img1, img2):
    ir_list = torch.split(img1, 1, dim=0)
    vis_list = torch.split(img2, 1, dim=0)
    fi_list = torch.split(fusion, 1, dim=0)

    ir_list_np = [ir_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(ir_list))]
    vis_list_np = [vis_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(vis_list))]
    fi_list_np = [fi_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(fi_list))]
    ssim_list = [
        np.mean((fi_list_np[ind]-ir_list_np[ind]) ** 2) + np.mean((fi_list_np[ind]-vis_list_np[ind]) ** 2) for
        ind in range(len(fi_list))]
    return ssim_list

def imresize(arr, size, interp='bilinear', mode=None):
    numpydata = np.asarray(arr)
    im = Image.fromarray(numpydata, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return np.array(imnew)


def resize(image1, image2, crop_size_img, crop_size_label):
    image1 = imresize(image1, crop_size_img, interp='bicubic')
    image2 = imresize(image2, crop_size_label, interp='bicubic')
    return image1, image2


def get_image_files(input_folder):
    valid_extensions = (".bmp", ".tif", ".jpg", ".jpeg", ".png")
    return sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)])

def calculate_weights_torch(IA: torch.Tensor, IB: torch.Tensor, tau: float, c: float, use_sign: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    批量计算显著图权重（PyTorch版本）

    参数:
        IA (torch.Tensor): 图像A的显著图，形状为 (B, 1, H, W) 或 (B, H, W)
        IB (torch.Tensor): 图像B的显著图，形状同IA
        tau (float): 显著性阈值
        c (float): 温度系数，控制权重分布的平滑性
        use_sign (bool): 是否显式使用torch.sign（默认False，推荐直接比较）

    返回:
        wA (torch.Tensor): 图像A的权重，形状为 (B,)
        wB (torch.Tensor): 图像B的权重，形状为 (B,)
    """
    # 确保输入形状为 (B, H, W)
    if IA.dim() == 4 and IA.shape[1] == 1:
        IA = IA.squeeze(1)  # 压缩通道维度 (B, 1, H, W) -> (B, H, W)
    if IB.dim() == 4 and IB.shape[1] == 1:
        IB = IB.squeeze(1)

    B, H, W = IA.shape

    # 公式8：计算显著图均值（支持批次）
    if use_sign:
        mask_A = (torch.sign(IA - tau) + 1) / 2  # 显式使用Sign函数
        mask_B = (torch.sign(IB - tau) + 1) / 2
    else:
        mask_A = (IA >= tau).float()  # 直接比较（推荐）
        mask_B = (IB >= tau).float()

    # 对每个样本计算均值，sum(dim=(1,2)) 对H和W求和
    mA_sal = torch.sum(mask_A * IA, dim=(1, 2)) / (H * W)  # 形状 (B,)
    mB_sal = torch.sum(mask_B * IB, dim=(1, 2)) / (H * W)  # 形状 (B,)

    # 公式9：Softmax计算权重（向量化实现）
    exp_A = torch.exp(mA_sal / c)
    exp_B = torch.exp(mB_sal / c)
    total = exp_A + exp_B
    wA = exp_A / total
    wB = exp_B / total

    return wA, wB

def rgb2ycbcr(rgb_image):
    H, W = rgb_image.shape[2], rgb_image.shape[3]
    device = rgb_image.device
    transform_matrix = torch.tensor([[0.257, 0.564, 0.098],
                                     [-0.148, -0.291, 0.439],
                                     [0.439, -0.368, -0.071]]).to(device)

    rgb_image = rgb_image.permute(0, 2, 3, 1).reshape(-1, 3)
    bias = torch.tensor([0.0625, 0.5, 0.5]).to(device)
    ycbcr_image = torch.matmul(rgb_image, transform_matrix.T) + bias

    ycbcr_image = ycbcr_image.reshape(1, H, W, 3).permute(0, 3, 1, 2)
    return ycbcr_image


def ycbcr2rgb(ycrcb_tensor):

    device = ycrcb_tensor.device
    H, W = ycrcb_tensor.size(2), ycrcb_tensor.size(3)

    transform_matrix = torch.tensor([[1.164, 0.000, 1.596],
                                     [1.164, -0.392, -0.813],
                                     [1.164, 2.017, 0.000]]).to(device)

    bias = torch.tensor([0.0625, 0.5, 0.5]).to(device)
    # 将YCRCB图像的通道维度调整为适合矩阵乘法的形状
    ycrcb_tensor = ycrcb_tensor.permute(0, 2, 3, 1).reshape(-1, 3)
    # 执行矩阵乘法
    rgb_tensor = torch.matmul(ycrcb_tensor-bias, transform_matrix.T)
    # 将结果重新调整为图像张量的形状
    rgb_tensor = rgb_tensor.reshape(-1, H, W, 3).permute(0, 3, 1, 2)

    return rgb_tensor