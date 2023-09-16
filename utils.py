# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 09:20:31 2022

@author: sunkg
"""
import torch.nn as nn
import math
import torch
import numpy as np
import torch.fft
import torch.nn.functional as F
import nibabel as nib
import h5py


# Filter out non-dummy images      
def check_img(img, label):
    imgs = []
    t1_exist = [idx for idx, t1 in enumerate(img) if t1.abs().sum() > 0]
    img_ = img[t1_exist, :] 
    imgs.append(img_)
    label_ = label[t1_exist, :]
    return imgs, label_
    

# Filter out non-dummy labels  
def check_label(img, label):
    label_exist = [idx for idx, label_ in enumerate(label) if label_ != -1e3]
    label_ = label[label_exist, :]
    img_ = [im[label_exist, :] for im in img] 
    return img_, label_


# Select image pair with at least one GT of auxiliary modality exists
def check_gen(img, label):
    exits = []
    for idx in range(len(img[-1])):
        for mod in range(len(img[:-1])):
            if img[mod][idx].abs().sum() != 0:
                exits.append(idx)
                break

    img_ = [im[exits, ...] for im in img] 
    label_ = label[exits, ...]
    
    return img_, label_


# Calculate weights for waCE based on ratio of amount of data between classes (negative samples against positive ones)
def cal_weights(imgs, label):
    weights = []
    for img in imgs:
        img_, label_ = check_img(img, label)
        weight_ = (len(label_) - torch.Tensor([i for i in label_]).sum())/max(torch.Tensor([i for i in label_]).sum(), 1e-3) 
        weights.append(weight_)
    return weights


# Drop the data if weight is larger than certain threshold
def aggressive_weights(weights, thresh):

    if np.array([1 if (i<=thresh) & (i>=1/thresh) else 0 for i in weights]).sum() == 0:
        return 1
    else:
        return 0


# Count samples of different classes 
def count_samples(data_list, protocals, lookup_table, mod = 'Train', augment = False):
  
    tl = [i[1] for i in data_list]
    if (augment == True) and (mod == 'Train'):
        n_tl1 = tl.count(0) + tl.count(2) # 0 and 2 for NC and sMCI
        n_tl2 = tl.count(1) + tl.count(3) # 1 and 3 for AD and pMCI
    else:
        n_tl1 = tl.count(lookup_table[protocals[0]])
        n_tl2 = tl.count(lookup_table[protocals[1]])
      
    return n_tl1, n_tl2


#### Loss L_B ####
def criteria_generator_bottleneck(x_gt, x_hat_fe, x_gt_fe, weight_g, backbone, device):
    L_MmD = torch.nn.MSELoss().to(device)
    
    if backbone == '3DCNN':
        L_cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    elif backbone == 'PyramidCNN':
        L_cos = nn.CosineSimilarity(dim = 1, eps = 1e-6)

    L_KL = nn.KLDivLoss(reduction="batchmean", log_target=True).to(device)
    Loss = torch.zeros([1]).to(device)

    for i in range(len(x_gt)):
        if x_gt[i].abs().sum() == 0:
            continue
        i_exist = [idx for idx, i_batch in enumerate(x_gt[i]) if i_batch.abs().sum()>0]

        hat = x_hat_fe[i][i_exist,...]
        gt = x_gt_fe[i][i_exist,...]

        if backbone == '3DCNN':
            B, C, W, H, D = gt.shape
            hat = hat.view(B, C, -1)
            gt = gt.view(B, C, -1)
                 
        #### MSE ####
        L1 = weight_g[0] * L_MmD(hat, gt)
        #### Cosine ####
        L2 = weight_g[1] * L_cos(hat, gt).mean()
        #### KL ####
        if backbone == '3DCNN':
            hat_log = F.log_softmax(hat, dim = 2)
            gt_log = F.log_softmax(gt, dim = 2)
        elif backbone == 'PyramidCNN':
            hat_log = F.log_softmax(hat, dim = 1)
            gt_log = F.log_softmax(gt, dim = 1)

        L3 = weight_g[2] * L_KL(hat_log, gt_log)

        Loss = Loss + L1 - L2 + L3

    return Loss


#### Loss L_H ####
def criteria_fe_cls(x_gt, x_hat_fe, x_gt_fe, device):
    # Loss function
    L_MmD = torch.nn.MSELoss().to(device)
    L_cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    Loss = torch.zeros([1]).to(device)

    for i in range(len(x_gt)):
        if x_gt[i].abs().sum() == 0:
            continue
        i_exist = [idx for idx, i_batch in enumerate(x_gt[i]) if i_batch.abs().sum()>0]

        hat = x_hat_fe[i][i_exist,...]
        gt = x_gt_fe[i][i_exist,...]
        #### MSE ####
        L1 = L_MmD(hat, gt)
        #### Cosine ####
        L2 = 5*L_cos(hat, gt).mean()

        Loss = Loss + L1 - L2 

    return Loss


#### Loss L_U1 (L_U for each branch) ####
def criteria_cls(x_gt, x_hat_cls, gt_cls, weight_g, device):
    # Loss function
    L_MmD = torch.nn.MSELoss().to(device)
    Loss = torch.zeros([1]).to(device)

    for i in range(len(x_gt)):
        if x_gt[i].abs().sum() == 0:
            continue
        i_exist = [idx for idx, i_batch in enumerate(x_gt[i]) if i_batch.abs().sum()>0]
        hat = x_hat_cls[i][i_exist,...]
        gt = gt_cls[i][i_exist,...]
        #### MSE ####
        L1 = weight_g[0]*L_MmD(hat, gt)
        Loss = Loss + L1 

    return Loss


#### Loss L_U2 (L_U for Decision integration) ####
def criteria_cls_merged(x_gt, x_hat_cls_merged, gt_cls_merged, weight_g, device):
    # Loss function
    L_MmD = torch.nn.MSELoss().to(device)
    Loss = torch.zeros([1]).to(device)

    for i in range(len(x_gt[-1])):
        flag = 0
        for j in range(len(x_gt)):
            if x_gt[j][i].abs().sum() == 0:
                flag = 1
                break
        if flag == 1:
            continue    
        #### MSE ####
        L1 = weight_g[0]*L_MmD(x_hat_cls_merged[i], gt_cls_merged[i])
        Loss = Loss + L1 

    return Loss


#### loss all ####
def criteria_joint(x_gt, x_hat_bottleneck, x_gt_bottleneck, x_hat_fe_cls, gt_fe_cls, x_hat_fe_cls_s, gt_fe_cls_s, x_hat_cls, gt_cls, x_hat_cls_merged, gt_cls_merged, Loss_classification, weight_g, weight_c, backbone, device):
    weight = 1e-2
    
    Loss_generator_bottleneck = criteria_generator_bottleneck(x_gt, x_hat_bottleneck, x_gt_bottleneck, weight_g, backbone, device)    
    Loss_fe_cls = criteria_fe_cls(x_gt, x_hat_fe_cls, gt_fe_cls, device)
    Loss_fe_cls_s = criteria_fe_cls(x_gt, x_hat_fe_cls_s, gt_fe_cls_s, device)
    Loss_cls = criteria_cls(x_gt, x_hat_cls, gt_cls, weight_g, device)
    Loss_cls_merged = criteria_cls_merged(x_gt, x_hat_cls_merged, gt_cls_merged, weight_g, device)
    Loss_classification = weight_c*Loss_classification
    Loss = Loss_generator_bottleneck + weight*Loss_fe_cls + weight*Loss_fe_cls_s + weight*Loss_cls + weight*Loss_cls_merged + Loss_classification

    return Loss, Loss_generator_bottleneck + weight*Loss_fe_cls + weight*Loss_fe_cls_s + weight*Loss_cls + weight*Loss_cls_merged, Loss_classification


def ssimloss(X, Y):
    win_size = 7
    k1 = 0.01
    k2 = 0.03
    w = torch.ones(1, 1, win_size, win_size).to(X) / win_size ** 2
    NP = win_size ** 2
    cov_norm = NP / (NP - 1)
    data_range = 1
    #data_range = data_range[:, None, None, None]
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    ux = F.conv2d(X, w)
    uy = F.conv2d(Y, w)
    uxx = F.conv2d(X * X, w)
    uyy = F.conv2d(Y * Y, w)
    uxy = F.conv2d(X * Y, w)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux ** 2 + uy ** 2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D
    return S.mean()


def lncc_loss(I, J, win=None):

    ndims = len(list(I.size())) - 2
    assert ndims ==  2, "volumes should be 2 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims


    sum_filt = torch.ones([1, 1, *win]).to(I)

    pad_no = math.floor(win[0]/2)

    stride = (1,1)
    padding = (pad_no, pad_no)
    
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross*cross / (I_var*J_var + 1e-5)

    return  torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


def compute_marginal_entropy(values, bins, sigma):
    normalizer_1d = np.sqrt(2.0 * np.pi) * sigma
    sigma = 2*sigma**2
    p = torch.exp(-((values - bins).pow(2).div(sigma))).div(normalizer_1d)
    p_n = p.mean(dim=1)
    p_n = p_n/(torch.sum(p_n) + 1e-10)
    return -(p_n * torch.log(p_n + 1e-10)).sum(), p


def gradient(x,h_x=None,w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow(torch.pow((r - l), 2) + torch.pow((t - b), 2), 0.5)
    return xgrad


def convert(nii_path, h5_path, protocal):
    h5 = h5py.File(h5_path, 'w')
    nii = nib.load(nii_path)
    array = nib.as_closest_canonical(nii).get_fdata() #convert to RAS
    array = array.T.astype(np.float32)
    h5.create_dataset('image', data=array)
    h5.attrs['max'] = array.max()
    h5.attrs['acquisition'] = protocal
    h5.close()


def Cosine(vec1, vec2):   
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()    
    return torch.dot(vec1, vec2.T)/(torch.linalg.norm(vec1) * torch.linalg.norm(vec2))


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1, bias = False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(N, in_chans, H, W, D)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W, D)`.
        """
        output = self.layers(image)
        return output


class ConvBlocks(nn.Module):
    def __init__(self, in_chans, out_chans, num_convs):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_convs = num_convs

        layers = [ConvBlock(in_chans if i == 0 else out_chans, out_chans) for i in range(self.num_convs)]
        self.layers = nn.Sequential(*layers)
        

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.num_convs != 0:
            output = self.layers(image)
        else:
            output = image
        return output
    

class DownConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(DownConvBlock, self).__init__()

        self.layers = nn.Sequential(
                                    nn.Conv3d(in_filters, out_filters, 3, 2, 1),
                                    nn.InstanceNorm3d(out_filters),
                                    nn.LeakyReLU(0.2, inplace=True)
                                    #nn.Dropout3d(0.25)
                                    )
    def forward(self, image):
        return self.layers(image)


class DownConvGen(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(DownConvGen, self).__init__()
        
        self.layers = nn.Sequential(
                                    nn.Conv3d(in_filters, out_filters, 3, 2, 1), 
                                    nn.InstanceNorm3d(out_filters),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv3d(out_filters, out_filters, 3, 1, 1),
                                    nn.InstanceNorm3d(out_filters),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    #nn.Dropout3d(0.25)
                                    )
    def forward(self, image):
        return self.layers(image)
    
    
class DownConvMiSe(nn.Module):
    def __init__(self, in_filters, out_filters, ks, stride, pad):
        super(DownConvMiSe, self).__init__()
        
        self.layers = nn.Sequential(
                                    nn.Conv3d(in_filters, out_filters, ks, stride, pad), 
                                    nn.BatchNorm3d(out_filters),
                                    nn.ReLU()
                                    )
    def forward(self, image):
        return self.layers(image)