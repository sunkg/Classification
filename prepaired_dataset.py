#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:30:44 2022

@author: sunkg
"""

import csv 
import os
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
import random
import math
import numbers
import torch.nn as nn
from torch.autograd import Variable
import glob
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBlur,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)

def build_labeldict(protocal, label_dict):
    labeldict_ = dict()
    for idx, p_ in enumerate(protocal):
        labeldict_[label_dict[p_]] = idx  
    return labeldict_

def select_label(protocal, label_dict):
    labeldict_ = dict()
    for idx, p_ in enumerate(protocal):
        labeldict_[p_] = label_dict[p_]   
    return labeldict_

def select_modality(protocal, mod_dict):
    select_mod = []
    for idx, p_ in enumerate(protocal):
        select_mod.append(mod_dict[p_])   
    return select_mod
    
def generate_lists(dataset, data_path, wholetable_path, group_size, label_dict, n_fold, num_repeat = None):
    
    dataset_lists = []
    with open(wholetable_path,'r') as labelfile:
        labels = csv.reader(labelfile)
        
        if dataset == 'ADNI':
            files = [[label[1], label_dict[label[7]]] for idx, label in enumerate(labels) if (idx != 0) & (label[7] != 'NAN') & (label[7] in label_dict)] 
        elif dataset == 'HS':
            files_all = [[label[0], label_dict[label[4]]] for idx, label in enumerate(labels) if (idx != 0) & (label[4] != 'None') & (label[4] in label_dict)]
            selected_files = [file for file in os.listdir(data_path)]
            files = [file for file in files_all if file[0].split('_')[0] in selected_files]
        elif dataset == 'RJ':
            files_all = [[label[0], label_dict[label[28]]] for idx, label in enumerate(labels) if (idx != 0) & (label[28] != '') & (label[0] != '') & (label[28] in label_dict)] 
            selected_files = [file for file in os.listdir(data_path)]
            files = []
            for file in files_all:
                for ff in selected_files:
                    if ff.startswith(file[0]):
                        files.append(file)
                        break
        elif dataset == 'OASIS':
            files, test_lists_ = [], []
            for idx, label in enumerate(labels):
                if (float(label[4]) + float(label[5]) > 0) & (label[10] in label_dict):
                    files.append([label[0]+'_'+label[1], label_dict[label[10]]])
                if (float(label[4]) + float(label[5]) == 0) & (label[10] in label_dict):           
                    test_lists_.append([label[0]+'_'+label[1], label_dict[label[10]]])
        elif dataset == 'BraTS':
            files = [[str(label[0]).rjust(5, '0'), label_dict[label[1]]] for idx, label in enumerate(labels) if (idx != 0)] 

        if n_fold != 0:
            fold_size = len(files)//n_fold

    #print(files)
    random.shuffle(files)    
    
    for fold_idx in range(n_fold):
        test_range_start = fold_idx * fold_size 
        test_lists = []

        if fold_idx != n_fold -1:
            test_list_ = files[test_range_start:test_range_start+fold_size]
            train_val_list = files[:test_range_start] + files[test_range_start+fold_size:]
        else:
            test_list_ = files[test_range_start:]
            train_val_list = files[:test_range_start]
        
        if dataset == 'OASIS':
            test_lists = test_lists + test_lists_
        else:
            test_lists = test_lists + test_list_
       
        random.shuffle(train_val_list)
        
        if n_fold == 10:
            train_list = train_val_list[:(n_fold-3)*fold_size]
            val_list = train_val_list[(n_fold-3)*fold_size:]
        elif n_fold == 5:
            train_list = train_val_list[:(n_fold-2)*fold_size]
            val_list = train_val_list[(n_fold-2)*fold_size:]
        else:
            raise RuntimeError(
                'Only 5- and 10-fold cross validation are supported.'
            )              
        
        if dataset == 'OASIS':
            train_list += test_list_

            
        if num_repeat != None:
            test_lists = test_lists + [file for _ in range(num_repeat-1) for file in test_list_]
        
        dataset_lists.append([train_list, val_list, test_lists])

    
    return dataset_lists, len(train_list)


def generate_groups(data_list, group_size, shuffle = True):
    data_lists = []
    if shuffle:
        random.shuffle(data_list)
    for i in range(0, group_size*(len(data_list)//group_size), group_size):
        data_lists.append(data_list[i: i+group_size])
        
    return data_lists
        
        
def pad_imgs(data, pads):
    data = F.pad(data, pad=pads, mode='constant', value=0)
    return data
     

def unpad_imgs(
    x: torch.Tensor,
    pads
) -> torch.Tensor:
    return x[...,pads[0]: x.shape[-3] - pads[1], pads[2] : x.shape[-2] - pads[3], pads[4] : x.shape[-1] - pads[5]]
  
          
def center_crop(data, shape):
    if shape[0] <= data.shape[-3]:
        w_from = (data.shape[-3] - shape[0]) // 2
        w_to = w_from + shape[0]
        data = data[..., w_from:w_to, :, :]
    else:
        w_before = (shape[0] - data.shape[-3]) // 2
        w_after = shape[0] - data.shape[-3] - w_before
        pad = [0, 0] * data.ndim
        pad[4:6] = [w_before, w_after]
        data = F.pad(data, pad=tuple(pad), mode='constant', value=0)
    if shape[1] <= data.shape[-2]:
        h_from = (data.shape[-2] - shape[1]) // 2
        h_to = h_from + shape[1]
        data = data[..., h_from:h_to, :]
    else:
        h_before = (shape[1] - data.shape[-2]) // 2
        h_after = shape[1] - data.shape[-2] - h_before
        pad = [0, 0] * data.ndim
        pad[2:4] = [h_before, h_after]
        data = F.pad(data, pad=tuple(pad), mode='constant', value=0)
    if shape[2] <= data.shape[-1]:
        d_from = (data.shape[-1] - shape[2]) // 2
        d_to = d_from + shape[2]
        data = data[..., d_from:d_to]
    else:
        d_before = (shape[2] - data.shape[-1]) // 2
        d_after = shape[2] - data.shape[-1] - d_before
        pad = [0, 0] * data.ndim
        pad[:2] = [d_before, d_after]
        data = F.pad(data, pad=tuple(pad), mode='constant', value=0)
    return data


def RandomGroup(path_csv, dataset_size):
    datalist = []
    with open(path_csv, 'r') as f:
        reader = csv.reader(f, delimiter = ",")
        data_ = list(reader)
        data_ = [i[0] for i in data_]
        random.shuffle(data_)
        for i in range(0, len(data_), dataset_size):
            datalist.append(data_[i: i+dataset_size])
    return datalist, len(data_)


def augment():
    training_transform = Compose([
    RandomNoise(std=0.01),
    RandomBlur(),
    OneOf({
        RandomAffine(scales = 0.03, degrees = 3),
        RandomElasticDeformation(max_displacement = 3)
    })
    ])
    return training_transform


def get_volumes(dataset, file_path, file_list, mod_paths, selected_labels, renamed_labels, crop = tuple([112, 128, 112]), pads = tuple([0, 0] * 3), mod_augment = False, data_augment = ['None'], n_mean = 0, n_std = 10, G_kernel = None, G_sigma = 1e-12):
    x_inputs = []
        
    for file, label in file_list:
        x_input, x_input_ = [], []
        for mod_path in mod_paths:
            
            if dataset in ['ADNI', 'HS', 'RJ', 'BraTS']:
                file_input = glob.glob(os.path.join(file_path, file, mod_path)) 
            elif dataset in ['OASIS']:
                file_input = glob.glob(os.path.join(file_path, file.split('_')[0], file.split('_')[1], mod_path, file + '.nii.gz'))

            if len(file_input) != 0:
                volume = nib.load(file_input[0]).get_fdata().astype('float32')
                volume = torch.from_numpy(volume).unsqueeze(0)
                volume = pad_imgs(volume, pads)
                volume_crop = center_crop(volume, crop) 

                if mod_path in data_augment: 
                    volume_crop = volume_crop.unsqueeze(0)
                    volume_blurry = AddBlur(volume_crop, G_kernel, G_sigma)
                    volume_crop = AddNoise(volume_blurry, n_mean, n_std)
                    volume_crop = volume_crop.squeeze(0)
            else:
                #print('Oooops!! Missing modality!', file, mod_path)
                volume_crop = np.zeros(crop, dtype='float32') #If data is missing, assign a dummy one
                volume_crop = torch.from_numpy(volume_crop).unsqueeze(0)
       
            if dataset in ['ADNI', 'HS', 'OASIS']:
                volume_crop = F.interpolate(volume_crop.unsqueeze(0), size=tuple([112, 128, 112])) 
                volume_crop = ImgNorm(volume_crop.squeeze(0))
            elif dataset in ['RJ', 'BraTS']:
                volume_crop = ImgNorm(volume_crop) 
            
            x_input.append(volume_crop)
        x_input_.append(x_input)

        if dataset == 'RJ':
            if label == 0 or label == 2: 
                label = 0 
        elif mod_augment == False:
            if label not in selected_labels.values():       
                label = -1e3 #If label is not demanded, assign a dummy one
            else:
                label = renamed_labels[label]
        elif (mod_augment == True) & (dataset == 'ADNI'): # To augment ADNI data: sMCI as NC, pMCI as AD
            if label == 0 or label == 2:
                label = 0
            elif label == 1 or label == 3:
                label = 1

        x_input_.append(Variable(torch.Tensor([label]).long()))
        x_inputs.append(x_input_)           
    return x_inputs


def ImgNorm(img):
    img = (img-img.mean())/(img.std()+1e-9)
    return img

def AddNoise(img, noise_mean = 0, noise_std = None):
    if noise_std != None:
        img_noisy = img + torch.randn(img.shape) * noise_std + noise_mean

    return img_noisy


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=3):
        super(GaussianSmoothing, self).__init__()
        sigma = max(sigma, 1e-12)
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
    
    
def AddBlur(img, kernelsize = None, sigma = 1e-12):
    
    if sigma <= 1:
        kernelsize = kernelsize or 3
        pad = 1
    elif sigma <= 2:
        kernelsize = kernelsize or 5
        pad = 2
    elif sigma > 2:
        kernelsize = kernelsize or 7
        pad = 3
    else:
        raise Exception("Not implemented!!!")

    smoothing = GaussianSmoothing(1, kernelsize, sigma) # channels = 1
        
    img = pad_imgs(img, tuple([pad, pad]*3))
    img = smoothing(img)
    return img


class VolumeDataset(torch.utils.data.Dataset):
    def __init__(self, volumes):
        super().__init__()
        self.volumes = volumes
    
    def __len__(self):
        return len(self.volumes)
    
    def __getitem__(self, index):
        volume = self.volumes[index]
        return volume

def generate_test_lists(dataset, data_path, wholetable_path, group_size, label_dict, n_fold, num_repeat = None):
    
    with open(wholetable_path,'r') as labelfile:
        labels = csv.reader(labelfile)
        
        if dataset == 'ADNI':
            files = [[label[1], label_dict[label[7]]] for idx, label in enumerate(labels) if (idx != 0) & (label[7] != 'NAN') & (label[7] in label_dict)] 
        elif dataset == 'HS':
            files_all = [[label[0], label_dict[label[4]]] for idx, label in enumerate(labels) if (idx != 0) & (label[4] != 'None') & (label[4] in label_dict)]
            selected_files = [file for file in os.listdir(data_path)]
            files = [file for file in files_all if file[0].split('_')[0] in selected_files]
        elif dataset == 'RJ':
            files_all = [[label[0], label_dict[label[47]]] for idx, label in enumerate(labels) if (idx != 0) & (label[47] != '') & (label[0] != '') & (label[47] in label_dict)] 
            selected_files = [file for file in os.listdir(data_path)]
            files = []
            for file in files_all:
                for ff in selected_files:
                    if ff.startswith(file[0]):
                        files.append(file)
                        break
        elif dataset == 'OASIS':
            files, test_lists_ = [], []
            for idx, label in enumerate(labels):
                if (float(label[4]) + float(label[5]) > 0) & (label[10] in label_dict):
                    files.append([label[0]+'_'+label[1], label_dict[label[10]]])
                if (float(label[4]) + float(label[5]) == 0) & (label[10] in label_dict):           
                    test_lists_.append([label[0]+'_'+label[1], label_dict[label[10]]])
        elif dataset == 'BraTS':
            files = [[str(label[0]).rjust(5, '0'), label_dict[label[1]]] for idx, label in enumerate(labels) if (idx != 0)] 

    #print(files)
    random.shuffle(files)    

    test_lists = []

    test_list_ = files[:]
    
    if dataset == 'OASIS':
        test_lists = test_list_ + test_lists_
    else:
        test_lists = test_lists + test_list_
                  
    if num_repeat != None:
        test_lists = test_lists + [file for _ in range(num_repeat-1) for file in test_list_]

    print('train_list length', len(test_lists))
    
    return test_lists


def get_testvolume(dataset, file_path, file_list, mod_paths, selected_labels, renamed_labels, pads = tuple([0, 0] * 3), n_mean = None, n_std = None, crop = tuple([112, 128, 112]), G_kernel = None, G_sigma = 1e-12, ratio = 1):

    file, label = file_list
    x_input, x_input_ = [], []
    
    #p = torch.rand(1)

    for mod_path in mod_paths:
        if dataset in ['ADNI', 'HS', 'RJ', 'BraTS']:
            file_input = glob.glob(os.path.join(file_path, file, mod_path)) 
        elif dataset in ['OASIS']:
            file_input = glob.glob(os.path.join(file_path, file.split('_')[0], file.split('_')[1], mod_path, file + '.nii.gz'))


        if len(file_input) != 0:
            volume = nib.load(file_input[0]).get_fdata().astype('float32')
            volume = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
            volume = pad_imgs(volume, pads)
            volume_crop = center_crop(volume, crop)
            
            #if p<ratio: 
            #    volume_blurry = AddBlur(volume_crop, G_kernel, G_sigma)
            #    volume_crop = AddNoise(volume_blurry, n_mean, n_std)
                               
        else:
            #print('Oooops!! Missing modality!', file, mod_path)
            volume_crop = np.zeros(crop, dtype='float32')# If missing, assign a dummy image
            volume_crop = torch.from_numpy(volume_crop).unsqueeze(0).unsqueeze(0) 
        
        if dataset in ['ADNI', 'OASIS', 'HS']:
            volume_crop = F.interpolate(volume_crop, size=tuple([112, 128, 112])) 
        
        volume_prepaired = ImgNorm(volume_crop) 
           
        x_input.append(volume_prepaired.cuda())
    x_input_.append(x_input)
    
    if label not in selected_labels.values():            
        label = -1e3
    else:
        label = renamed_labels[label]
    x_input_.append(Variable(torch.Tensor([[label]]).long()).cuda())
    return x_input_


def loader_data(dataset, file_path, file_list, mod_path, crop, batch_size, selected_labels, renamed_labels, pads, num_workers, shuffle, mod_augment = False, data_augment = ['None']):
    volumes = get_volumes(dataset, file_path, file_list, mod_path, selected_labels, renamed_labels, crop, pads, mod_augment, data_augment)  
    volumes = VolumeDataset(volumes)
    data_loader = torch.utils.data.DataLoader( \
            volumes, batch_size=batch_size, shuffle=shuffle, \
            num_workers=num_workers, pin_memory=True, drop_last=True)
    
    return data_loader
