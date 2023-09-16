#!/usr/bin/env python3

import os, os.path
import torch
import numpy as np
from prepaired_dataset import  select_label, build_labeldict, select_modality, get_testvolume
import model
import metrics      
import random
import csv 

def generate_test_lists(dataset, data_path, wholetable_path, group_size, label_dict, n_fold, num_repeat = None):
    
    with open(wholetable_path,'r') as labelfile:
        labels = csv.reader(labelfile)
        
        if dataset == 'ADNI':
            files_all = [[label[1], label_dict[label[7]]] for idx, label in enumerate(labels) if (idx != 0) & (label[7] != 'NAN') & (label[7] in label_dict)] 
            selected_files = [file for file in os.listdir(data_path)]
            files = [file for file in files_all if file[0] in selected_files]
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
            selected_files = [file for file in os.listdir(data_path)]
            for idx, label in enumerate(labels):
                if (float(label[4]) + float(label[5]) > 0) & (label[10] in label_dict):
                    if label[0] in selected_files:
                        if label[1] in os.listdir(os.path.join(data_path, label[0])):
                            files.append([label[0]+'_'+label[1], label_dict[label[10]]])
                if (float(label[4]) + float(label[5]) == 0) & (label[10] in label_dict):     
                    if label[0] in selected_files:
                        if label[1] in os.listdir(os.path.join(data_path, label[0])):
                            test_lists_.append([label[0]+'_'+label[1], label_dict[label[10]]])
        elif dataset == 'BraTS':
            files_all = [[str(label[0]).rjust(5, '0'), label_dict[label[1]]] for idx, label in enumerate(labels) if (idx != 0)] 
            selected_files = [file for file in os.listdir(data_path)]
            files = [file for file in files_all if file[0] in selected_files]

    random.shuffle(files)    

    test_lists = []

    test_list_ = files[:]
    
    if dataset == 'OASIS':
        test_lists = test_list_ + test_lists_
    else:
        test_lists = test_lists + test_list_
                  
    if num_repeat != None:
        test_lists = test_lists + [file for _ in range(num_repeat-1) for file in test_list_]
    
    return test_lists



def main(args):
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"]= ':4096:8'
    seed = 166021280
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for path in [args.save_path, args.save_path+'/ckpt', args.save_path+'/ckpt/C-Network', args.save_path+'/ckpt/J-Network']:
        if not os.path.exists(path):
            os.mkdir(path)
            print('mkdir:', path)
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    table_csv = args.table_csv
    
    if args.dataset == 'ADNI': # NC vs. AD, sMCI vs. pMCI
        args.lookup_table = {'CN': 0, 'AD': 1, 'sMCI': 2, 'pMCI': 3}
        args.mod_table = {'MRI': 'T1_brain*', 'FDG': 'FDG_smoothed*', 'AV45': 'AV45_smoothed*'}
        args.img_size = tuple([112, 128, 112])
        args.pads = tuple([0,0,0,0,0,0])
    elif args.dataset == 'OASIS': # NC vs. AD
        args.lookup_table = {'CN': 0, 'AD': 1}
        args.mod_table = {'MRI': 't1', 'FDG': 'fdg', 'AV45':'av45'}
        args.img_size = tuple([168, 192, 168])
        args.pads = tuple([0,0,0,0,0,0])
        args.protocals = ['CN','AD']
    elif args.dataset == 'HS': # NC vs. AD
        args.lookup_table = {'NC': 0, 'MCI': 1, 'AD': 2, 'sMCI': 3, 'pMCI': 4, 'None': 5}
        args.mod_table = {'MRI': 'MRI/U8t1_mprage*', 'FDG': 'FDG/U8PET_Brain_iterative*', 'AV45': 'AV45/U8PET_Brain_iterative*'}
        args.img_size = tuple([181, 207, 181])
        args.pads = tuple([0,0,0,0,0,0])
        args.protocals = ['NC','AD']
    elif args.dataset == 'RJ': # NCI vs. svMCI
        args.lookup_table = {'NCI': 0, 'na-MCI': 1, 'a-MCI': 1, 'a&md-MCI': 1, 'na&md-MCI': 1, 'HC': 2, 'Dementia': 3, 'AD': 4, 'None': 5}
        args.mod_table = {'MRI': 'T1*', 'Flair': 'Flair*'}
        args.img_size = tuple([176, 208, 176])
        args.protocals = ['NCI','a-MCI']
    elif args.dataset == 'BraTS': # MGMT- vs. MGMT+
        args.lookup_table = {'0': 0, '1': 1}
        args.mod_table = {'T1': '*t1.*', 'T1CE': '*t1ce*', 'T2':'*t2*','Flair':'*flair*'}
        args.img_size = tuple([192, 192, 144])
        args.pads = tuple([0,11,0,0,0,0])
        args.protocals = ['0','1']
    
    
    renamed_dict = build_labeldict(args.protocals, args.lookup_table)
    selected_dict = select_label(args.protocals, args.lookup_table)
    mod_dict = select_modality(args.modalities, args.mod_table)
    Metrics_folds = []

    test_list = generate_test_lists(args.dataset, args.test_path, table_csv, args.datablocksize, args.lookup_table, args.n_fold, args.n_repeat) 
    

    Predicted_labels, GT_labels, Uncertainty = [], [], [] 
    AUC, ACC, SPE, SEN, F1, Metrics = [], [], [], [], [], []

    if os.path.isfile(args.model_path) or os.path.isdir(args.model_path):
        ckpt = torch.load(args.model_path + '/Fold_'+str(args.fold)+'_best.pt')
        cfg = ckpt['config']
        if args.dataset in ['ADNI', 'OASIS', 'HS']:
            img_size = tuple([112, 128, 112]) 
        elif args.dataset in ['RJ', 'BraTS']:
            img_size = cfg.img_size

        net = model.PreUN(args.dataset, img_size, cfg.chans_gen, cfg.stages_gen, len(cfg.modalities), cfg.classes, cfg.lambda_epochs, cfg.shared_cls, cfg.shared_gen, backbone = cfg.backbone).to(device)
        net.load_state_dict(ckpt['state_dict'])
        print('load ckpt from:', args.model_path + '/Fold_'+str(args.fold)+'_best.pt')
    else:
        raise FileNotFoundError
        
    net.use_amp = False
    net.GT = args.GT
    net.eval()
    
    testl = [i[1] for i in test_list]
    n_testl1 = testl.count(args.lookup_table[args.protocals[0]])  
    n_testl2 = testl.count(args.lookup_table[args.protocals[1]])              
        
    print('test size %d(%d/%d)'%(len(test_list), n_testl1, n_testl2))

    for ratio_idx, ratio in enumerate(args.n_ratio):      
        for std_idx, n_std in enumerate(args.n_std):
            for sigma_idx, G_sigma in enumerate(args.G_sigma):
                Predicted_labels, Predicted_labels_all, GT_labels, Uncertainty = [], [], [], []
                features_, labels_, features_trans_, features_gt_, labels_trans_, views_, evidence_a_, features_hat_cls_, features_hat_cls_s_, features_gt_cls_, features_gt_cls_s_ = [], [], [], [], [], [], [], [], [], [], []
    
                
                for index_test_data, test_data in enumerate(test_list):            
                    x_input, label_gt = get_testvolume(args.dataset, args.test_path, test_data, mod_dict, selected_dict, renamed_dict, args.pads, n_mean = args.n_mean, n_std = n_std, crop = args.img_size, G_kernel = args.G_kernel, G_sigma = G_sigma, ratio = ratio) 

                    with torch.no_grad():   
                        if (x_input[-1].abs().sum() == 0) or (label_gt == -1e3): 
                            continue
                        
                        samples, trans, evidence_acc, x_hat_cls, x_hat_fe_cls, x_hat_fe_cls_s, x_hat_cls_merged, loss_MoC, predicted, uncertainty = net(x_input[-1], 30, 15, [1 for _ in range(len(x_input))], label_gt)        
                        _, _, x_gt_fe, _, _, _, views, gt_fe_cls, gt_fe_cls_s = net.Classifier(x_input, label_gt, 30, 15, [1 for _ in range(len(x_input))])
                        _, predicted_all = torch.max(torch.stack(x_hat_cls, dim = 1), 2)

                        Predicted_labels.append(predicted.cpu().numpy())
                        GT_labels.append(label_gt.cpu().numpy())
                        Uncertainty.append(uncertainty.cpu().numpy())
                        features_.append(samples)
                        features_hat_cls_.append(x_hat_fe_cls)
                        features_hat_cls_s_.append(x_hat_fe_cls_s)
                        labels_.append(np.arange(len(samples)))
                        features_gt_.append(x_gt_fe)
                        features_gt_cls_.append(gt_fe_cls)
                        features_gt_cls_s_.append(gt_fe_cls_s)
                        features_trans_.append(trans)
                        labels_trans_.append(label_gt.item() * np.ones(len(trans)))
                        views_.append(views)
                        Predicted_labels_all.append(predicted_all.tolist()[0])
                        evidence_a_.append(evidence_acc[0])   
                        
                        print('Predicted/GT/Uncertainty:%d/%d/%.4f' %(Predicted_labels[-1], GT_labels[-1], Uncertainty[-1]))

                
                Predicted_labels = np.array(Predicted_labels)
                Predicted_labels = Predicted_labels.flatten()
                GT_labels = np.array(GT_labels)
                GT_labels = GT_labels.flatten()   
                Uncertainty = np.array(Uncertainty)
                Uncertainty = Uncertainty.flatten()
                Predicted_labels_all = np.array(Predicted_labels_all)


                class_metrics = metrics.class_metrics(Predicted_labels, GT_labels, evidence_a_)
                auc = class_metrics[0]
                acc = class_metrics[1]   
                sen = class_metrics[2]   
                spe = class_metrics[3] 
                f1= class_metrics[4] 

                ACC.append(acc)
                AUC.append(auc)  
                SPE.append(spe)
                SEN.append(sen)
                F1.append(f1) 
                Metrics.append(class_metrics)

    np.save(args.save_path+'/ckpt/J-Network/fold'+str(args.fold)+'_ACC', np.array(ACC))
    np.save(args.save_path+'/ckpt/J-Network/fold'+str(args.fold)+'_AUC', np.array(AUC))
    np.save(args.save_path+'/ckpt/J-Network/fold'+str(args.fold)+'_SPE', np.array(SPE))
    np.save(args.save_path+'/ckpt/J-Network/fold'+str(args.fold)+'_SEN', np.array(SEN))
    Metrics_folds.append(Metrics)
     

if __name__ == '__main__':
    import argparse
   
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unknown boolean value.')
            
    parser = argparse.ArgumentParser(description='PreUN')
    parser.add_argument('--model_path', type=str, default='/home/sunkg/Projects/SynthesisPET/models/ckpt/C-Network/best.pt', \
                        help='with ckpt path, C-Network or S-Network or J-Network')
    parser.add_argument('--save_path', default = '/home/sunkg/Projects/SynthesisPET/models/results', \
                        type=str, help='path to save evaluated data')
    parser.add_argument('--save_img', action='store_true', \
                        help='save images or not')
    parser.add_argument('--test_path', default='/home/sunkg/Projects/SynthesisPET/Data/PETCenter/DataALL', \
                        type=str, help='path to csv of test data')
    parser.add_argument('--test_mode', type=str, default='C', choices=['C','J'], \
                        help='Model to test, C (Classification), S (Synthesis) or J (Joint)')
    parser.add_argument('--table_csv', type=str, default='/home/sunkg/Projects/SynthesisPET/Data/ADNIALL/ADNI.csv', \
                        help='path to csv file of the scan record')
    parser.add_argument('--fold', type=int, default=1, \
                            help='fold index, from 1 to 5')
    parser.add_argument('--GT', type=bool, default=True, \
                        help='if there is GT, default is True') 
    parser.add_argument('--pads', type=tuple, default= tuple([0,0,0,0,0,0]), \
                        help='size of each image dimension')   
    parser.add_argument('--img_size', type=tuple, default=tuple([112, 128, 112]), \
                        help='size of each image dimension')
    parser.add_argument('--dataset', type=str, default='ADNI',  \
                        choices=['ADNI', 'HS', 'OASIS', 'RJ', 'BraTS'], \
                        help='dataset to choose')
    parser.add_argument('--backbone', type=str, default='3DCNN', \
                        choices= ['3DCNN','PyramidCNN'], help='Backbone to choose')
    parser.add_argument('--modalities', metavar='NAME', \
                        type=str, default= ['FDG', 'AV45', 'MRI'], nargs='*', \
                        help='Modalities to synthesize')
    parser.add_argument('--datablocksize', type=int, default=10, \
                            help='block size of data')
    parser.add_argument('--uncertainty', type=str2bool, default=True, \
                        help='use uncertainty, default is true') 
    parser.add_argument('--protocals', metavar='NAME', \
                        type=str, default= ['CN','AD'], nargs='*', help='Groups to classify')
    parser.add_argument('--n_mean', type=float, default=None, \
                        help='Mean of additive Gaussian noise, default is None') 
    parser.add_argument('--n_std', type=tuple, default=[0], \
                        help='Std of additive Gaussian noise, default is None') 
    parser.add_argument('--n_ratio', type=tuple, default=[0], \
                        help='ratio of data that is contaminated') 
    parser.add_argument('--G_sigma', type=tuple, default=[1e-12], \
                        help='Gaussian blur kernel, default is 1e-12') 
    parser.add_argument('--G_kernel', type=int, default=None, \
                        help='Gaussian kernel size, default is 5') 
    parser.add_argument('--n_repeat', type=int, default=1, \
                    help='Repeated time of test data, default is None') 
    parser.add_argument('--n_fold', type=int, default=5, \
                        help='XX-fold cross validation') 
    args = parser.parse_args()

    main(args)

