import argparse
import os
import numpy as np
import metrics
from prepaired_dataset import loader_data, generate_lists, generate_groups, select_label, build_labeldict, get_testvolume, select_modality
import model
import torch
import random
from utils import criteria_joint, check_img, check_label, check_gen, cal_weights, aggressive_weights, count_samples


  
def Instantiate(device, Load_CNetwork = None, Load_JNetwork = None, dataset = 'ADNI', backbone = '3DCNN'):
    if opt.train == 'C': 
        if dataset in ['ADNI', 'OASIS', 'HS']:
            img_size = tuple([112, 128, 112]) 
        else:
            img_size = opt.img_size #RenJi, BraTS

        # Initialize Classifier
        if Load_CNetwork != None:
            ckpt_C = torch.load(Load_CNetwork)
            cfg = ckpt_C['config']
            if opt.uncertainty == True:
                #### Classifier with uncertainty ####
                net = model.Classifier(dataset, img_size, len(cfg.modalities), cfg.classes, cfg.lambda_epochs, opt.shared_cls, depth = opt.stages_gen, chans = opt.chans_gen, backbone = backbone).to(device)
            else:
                #### Classifier without uncertainty ####
                if backbone == '3DCNN':
                    net = model.StandardClassifier(img_size, len(cfg.modalities), cfg.classes, cfg.shared_cls, depth = opt.stages_gen, chans = opt.chans_gen).to(device)
                elif backbone == 'PyramidCNN':
                    net = model.StandardMiSePy(dataset, img_size, len(opt.modalities), opt.classes, opt.chans_gen).to(device)
            net.load_state_dict(ckpt_C['state_dict'])
            print('Load pretrained from: ', Load_CNetwork)
        else:
            if opt.uncertainty == True:
                #### Classifier with uncertainty ####
                net = model.Classifier(dataset, img_size, len(opt.modalities), opt.classes, opt.lambda_epochs, opt.shared_cls, depth = opt.stages_gen, chans = opt.chans_gen, backbone = backbone).to(device)
                print('Use uncertainty-based classifier.')
            else:
                #### Classifier without uncertainty ####
                if backbone == '3DCNN':
                    net = model.StandardClassifier(img_size, len(opt.modalities), opt.classes, opt.shared_cls, depth = opt.stages_gen, chans = opt.chans_gen).to(device)
                elif backbone == 'PyramidCNN':
                    net = model.StandardMiSePy(dataset, img_size, len(opt.modalities), opt.classes, opt.chans_gen).to(device)
                print('Use softmax-based standard classifier.')
            print('Train from the scratch!')

        optimizer = torch.optim.Adam(net.parameters(), betas = (opt.b1, opt.b2), lr=opt.lr)
        #optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[opt.milestone], gamma=opt.decay_gamma)
        total = sum([param.nelement() for param in net.parameters()])
        print('C-Network size is %.2f M' % (total/1e6))   
        
    elif opt.train == 'J':
        if dataset in ['ADNI', 'OASIS', 'HS']:
            img_size = tuple([112, 128, 112])
        else:
            img_size = opt.img_size

        if Load_JNetwork != None:
            print('Train pretrained J-Network!')
            ckpt_J = torch.load(Load_JNetwork)
            cfg = ckpt_J['config']
            net = model.PreUN(dataset, img_size, cfg.chans_gen, cfg.stages_gen, len(cfg.modalities), cfg.classes, cfg.lambda_epochs, opt.shared_cls, opt.shared_gen, backbone = backbone).to(device)
            net.load_state_dict(ckpt_J['state_dict'])
            if Load_CNetwork == None:
               print('C-network is not additionally loaded.')
            elif os.path.isfile(Load_CNetwork):
                ckpt_C = torch.load(Load_CNetwork)
                net.Classifier.load_state_dict(ckpt_C['state_dict'])
            else:
                raise Exception("C-network not found!")
            print('Load pretrained from: ', Load_JNetwork, Load_CNetwork)
        
        else:
            print('Train J-Network from scratch!')
            if Load_CNetwork == None:
               raise Exception("C-network is required!")
            elif os.path.isfile(Load_CNetwork):
                ckpt_C = torch.load(Load_CNetwork)
                cfg = ckpt_C['config']
                net = model.PreUN(dataset, img_size, cfg.chans_gen, cfg.stages_gen, len(cfg.modalities), cfg.classes, opt.lambda_epochs, cfg.shared_cls, cfg.shared_gen, backbone = backbone).to(device)
                opt.shared_cls = cfg.shared_cls
                print('Loaded model:', Load_CNetwork)
                net.Classifier.load_state_dict(ckpt_C['state_dict'])

            else:
                raise Exception("C-network not found!")
            print('Load pretrained from: ', Load_CNetwork)

        optimizer = torch.optim.Adam(net.parameters(), betas=(opt.b1, opt.b2), lr=opt.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[opt.milestone], gamma=opt.decay_gamma)
        total = sum([param.nelement() for param in net.parameters()])
        print('J-Network size is %.2f M' % (total/1e6))
    return net, optimizer, scheduler
        
        
###################### MAIN ###################
def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if opt.dataset == 'ADNI': # NC vs. AD, sMCI vs. pMCI
        opt.lookup_table = {'CN': 0, 'AD': 1, 'sMCI': 2, 'pMCI': 3} 
        opt.mod_table = {'MRI': 'T1_brain*', 'FDG': 'FDG_smoothed*', 'AV45': 'AV45_smoothed*'}
        opt.img_size = tuple([112, 128, 112])
        opt.pads = tuple([0,0,0,0,0,0])
    elif opt.dataset == 'OASIS': # NC vs. AD
        opt.lookup_table = {'CN': 0, 'AD': 1}
        opt.mod_table = {'MRI': 't1', 'FDG': 'fdg', 'AV45':'av45'}
        opt.img_size = tuple([168, 192, 168])
        opt.pads = tuple([0,0,0,0,0,0])
        opt.protocals = ['CN','AD']
    elif opt.dataset == 'HS': # NC vs. AD
        opt.lookup_table = {'NC': 0, 'MCI': 1, 'AD': 2, 'sMCI': 3, 'pMCI': 4, 'None': 5} 
        opt.mod_table = {'MRI': 'MRI/U8t1_mprage*', 'FDG': 'FDG/U8PET_Brain_iterative*', 'AV45': 'AV45/U8PET_Brain_iterative*'}
        opt.img_size = tuple([181, 207, 181])
        opt.pads = tuple([0,0,0,0,0,0])
        opt.protocals = ['NC','AD']
    elif opt.dataset == 'RJ': # NCI vs. svMCI
        opt.lookup_table = {'NCI': 0, 'na-MCI': 1, 'a-MCI': 1, 'a&md-MCI': 1, 'na&md-MCI': 1, 'HC': 2, 'Dementia': 3, 'AD': 4, 'None': 5} 
        opt.mod_table = {'MRI': 'T1*', 'Flair': 'Flair*'}   
        opt.img_size = tuple([176, 208, 176])
        opt.protocals = ['NCI','a-MCI']
    elif opt.dataset == 'BraTS': # MGMT- vs. MGMT+
        opt.lookup_table = {'0': 0, '1': 1}
        opt.mod_table = {'T1': '*t1.*', 'T1CE': '*t1ce*', 'T2':'*t2*','Flair':'*flair*'}
        opt.img_size = tuple([192, 192, 144])
        opt.pads = tuple([0,11,0,0,0,0])
        opt.protocals = ['0','1']
    
    
    opt.renamed_dict = build_labeldict(opt.protocals, opt.lookup_table)
    opt.selected_dict = select_label(opt.protocals, opt.lookup_table)
    opt.mod_dict = select_modality(opt.modalities, opt.mod_table)
    opt.classes = len(opt.protocals)
    Metrics_folds = []
    os.environ["CUBLAS_WORKSPACE_CONFIG"]= ':4096:8'
    #torch.use_deterministic_algorithms(True)
    seed = 166021280
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(opt)
    print('Using seed',seed)
    print('Selected_dict', opt.selected_dict, 'Renamed_dict', opt.renamed_dict)
    
    for path in [opt.logdir, opt.logdir+'/ckpt', opt.logdir+'/ckpt/C-Network', opt.logdir+'/ckpt/J-Network']:
        if not os.path.exists(path):
            os.mkdir(path)
            print('mkdir:', path)
        
    if opt.dataset == 'RJ':
       dataset_lists, num_train = generate_lists(opt.dataset, opt.data_path, opt.table_csv, opt.datablocksize, dict(list(opt.lookup_table.items())[:6]), opt.n_fold, opt.n_repeat) # for multiple labeled data, RenJI 
    elif opt.mod_augment == False:
        dataset_lists, num_train = generate_lists(opt.dataset, opt.data_path, opt.table_csv, opt.datablocksize, opt.selected_dict, opt.n_fold, opt.n_repeat) 
    elif (opt.mod_augment == True) & (opt.dataset == 'ADNI'):
        dataset_lists, num_train = generate_lists(opt.dataset, opt.data_path, opt.table_csv, opt.datablocksize, opt.lookup_table, opt.n_fold, opt.n_repeat) # for augmented label only implemented for ADNI

   
    for fold_idx in range(opt.n_fold):

        print ('Training fold %d/%d ...'%(fold_idx+1, opt.n_fold))
        train_list, val_list, test_list = dataset_lists[fold_idx]
        n_tl1, n_tl2 = count_samples(train_list, opt.protocals, opt.lookup_table, mod = 'Train', augment = opt.mod_augment)
        n_vl1, n_vl2 = count_samples(val_list, opt.protocals, opt.lookup_table)
        n_testl1, n_testl2 = count_samples(test_list, opt.protocals, opt.lookup_table)
        print('Training datasize %d(%d/%d), validation size %d(%d/%d), test size %d(%d/%d)'%(len(train_list), n_tl1, n_tl2, n_vl1+n_vl2, n_vl1, n_vl2, n_testl1+n_testl2, n_testl1, n_testl2))

        print('test_list', fold_idx, test_list)
        
        AUC_best, ACC_best, SEN_best, SPE_best, F1_best = None, None, None, None, None
        train_loss = np.zeros([opt.n_epochs * num_train//opt.batch_size, 1])
        val_ACC = np.zeros([opt.n_epochs, 1])
        val_loss = np.zeros([opt.n_epochs, 1])
        val_loss_c = np.zeros([opt.n_epochs, 1])
        iter_cnt = 0
        metrics_old = np.zeros([6, 1])

        # Instantiate
        if opt.train == 'J':
            if opt.Load_CNetwork != None:
                fold_id = opt.fold_C or fold_idx+1
                Load_CNetwork = os.path.join(opt.Load_CNetwork, 'Fold_'+str(fold_id)+'_best.pt')
            else:
                Load_CNetwork = None
            if opt.Load_JNetwork != None:
                fold_id = opt.fold_J or fold_idx+1
                Load_JNetwork = os.path.join(opt.Load_JNetwork, 'Fold_'+str(fold_id)+'_best.pt')
                net, optimizer, scheduler = Instantiate(device, Load_CNetwork, Load_JNetwork, dataset = opt.dataset, backbone = opt.backbone)
            else:
                net, optimizer, scheduler = Instantiate(device, Load_CNetwork, dataset = opt.dataset, backbone = opt.backbone)
        elif opt.train == 'C':
            if opt.Load_CNetwork != None:
                fold_id = opt.fold_C or fold_idx+1
                Load_CNetwork = os.path.join(opt.Load_CNetwork, 'Fold_'+str(fold_id)+'_best.pt')
                net, optimizer, scheduler = Instantiate(device, Load_CNetwork, dataset = opt.dataset, backbone = opt.backbone)
            else:
                net, optimizer, scheduler = Instantiate(device, dataset = opt.dataset, backbone = opt.backbone)

           
        for epoch in range(opt.n_epochs):     
            
            train_groups = generate_groups(train_list, opt.datablocksize) 
            val_groups = generate_groups(val_list, 2, shuffle = False) 
            
                   
            ########################## Training ###############################
            net.train()

            features_, labels_, features_trans_, features_gt_, labels_trans_ = [], [], [], [], []
            for idx_list, train_list_ in enumerate(train_groups):
                
                dataloader_train = loader_data(opt.dataset,
                                               opt.data_path, 
                                               train_list_,
                                               opt.mod_dict, 
                                               opt.img_size, 
                                               batch_size = opt.batch_size,
                                               selected_labels = opt.selected_dict,
                                               renamed_labels = opt.renamed_dict,
                                               pads = opt.pads,
                                               num_workers = opt.num_workers,
                                               shuffle = True,
                                               mod_augment = opt.mod_augment,
                                               data_augment = opt.data_augment)

                for idx, batch in enumerate(dataloader_train):                    
                    x_input, label_gt = [x.to(device) for x in batch[0]], batch[1].to(device)  # last element of x_input is T1w  
                    
                    #### Training Stage I ####           
                    if opt.train == 'C':
                        #### ignore the data with dummy labels ####
                        x_input, label_gt = check_label(x_input, label_gt)
                                             
                        if (label_gt.shape[0] == 0):
                            continue    
                        #### weighted loss ####
                        weight = cal_weights(x_input, label_gt)

                        optimizer.zero_grad()

                        if opt.uncertainty == True:
                            #### Classifier with uncertainty ####   
                            if (label_gt.shape[0] == 0) or aggressive_weights(weight, 2):  # if ratio is larger than 2, then drop
                                continue
                            _, _, _, C_loss, predicted, uncertainty, _, _, _ = net(x_input, label_gt, epoch, opt.milestone, weight)
                       
                        else:
                            #### Single modality classifier without uncertainty ####
                            if  (weight[-1] > 2) or (weight[-1] < 1/2): # 2 is the threshold for data imbalance
                                continue
                            x_input, label_gt = check_img(x_input[0], label_gt)
                            predicted, C_loss, probs, _ = net(x_input, label_gt, weight[-1])
                        C_loss.backward()

                        optimizer.step()   
                                     
                        train_loss[iter_cnt] = C_loss.detach().cpu().numpy()
                                   
                    #### Training Stage II ####
                    elif opt.train == 'J': 
                        #### ignore the data with dummy labels ####
                        x_input, label_gt = check_label(x_input, label_gt)
                    
                        if (x_input[-1].abs().sum() == 0) or (label_gt.shape[0] == 0): 
                            continue 
                        
                        #### freeze the pretrained part from S1 ####
                        for param in net.Classifier.parameters():
                            param.requires_grad = False
                    
                        optimizer.zero_grad()
                        
                        #### Get weights for waCE ####
                        weight = cal_weights(x_input, label_gt)

                        x_input, label_gt = check_gen(x_input, label_gt)
                        if (label_gt.shape[0] == 0) or (aggressive_weights(weight[:-1], 5)):
                            continue

                        samples, x_hat_bottleneck, evidence_acc, x_hat_cls, x_hat_fe_cls, x_hat_fe_cls_s, x_hat_cls_merged, loss_MoC, predicted, uncertainty = net(x_input[-1], epoch, opt.milestone, weight, label_gt)        
                        gt_cls, gt_cls_merged, gt_bottleneck, gt_fe_cls, gt_fe_cls_s = net.Classifier.ClassifierBlock(x_input)
                                             
                        weight_g = opt.weight_g
                        weight_c = opt.weight_c

                        #### Calculate loss ####
                        J_loss, G_loss, C_loss = criteria_joint(x_input[:-1], x_hat_bottleneck, gt_bottleneck, x_hat_fe_cls, gt_fe_cls, x_hat_fe_cls_s, gt_fe_cls_s, \
                                                                x_hat_cls, gt_cls, x_hat_cls_merged, gt_cls_merged, loss_MoC, weight_g, weight_c, opt.backbone, device)   
                        

                        J_loss.backward()           
                        optimizer.step()   

                        train_loss[iter_cnt] = J_loss.detach().cpu().numpy()    
                        
                    iter_cnt += 1
            

            ###################  validation  ########################
            print('Validation at fold %d/%d, epoch %d'%(fold_idx+1, opt.n_fold, epoch+1))        

            net.eval()
            if opt.train == 'C':                    
                Predicted_labels, Predicted_labels_all, evidence_, loss_current, GT_labels = [], [], [], [], []
                           
                for val_list_ in val_groups:
                    dataloader_val = loader_data(opt.dataset,
                                                 opt.data_path, 
                                                 val_list_,
                                                 opt.mod_dict, 
                                                 opt.img_size, 
                                                 batch_size = 1,
                                                 selected_labels = opt.selected_dict,
                                                 renamed_labels = opt.renamed_dict,
                                                 pads = opt.pads,
                                                 num_workers = opt.num_workers,
                                                 shuffle=False)                
                    for i, batch in enumerate(dataloader_val):                      
                        x_input, label_gt = [x.to(device) for x in batch[0]], batch[1].to(device)                   
                        x_input, label_gt = check_label(x_input, label_gt)
                        if label_gt.shape[0] == 0:       
                            continue   
                        with torch.no_grad():
                            if opt.uncertainty == True:
                                #### Classifier with uncertainty ####
                                evidence, evidence_a, _, loss, predicted, uncertainty, _, _, _ = net(x_input, label_gt, opt.n_epochs, opt.milestone, [1 for _ in range(len(x_input))])
                                evidence_.append(evidence_a[0])
                                _, predicted_all = torch.max(torch.stack(evidence, dim = 1), 2)
                            else:
                                #### Single modality classifier without uncertainty ####
                                predicted, loss, output, features = net(x_input, label_gt, 1)
                                predicted_all = predicted.reshape(1,1)
                                evidence_.append(output)

                            Predicted_labels.append(predicted.cpu().numpy())
                            Predicted_labels_all.append(predicted_all.tolist()[0])
                            GT_labels.append(label_gt.cpu().numpy())
                            loss_current.append(loss.cpu())
                           
                Predicted_labels = np.array(Predicted_labels)
                Predicted_labels = Predicted_labels.flatten()
                Predicted_labels_all = np.array(Predicted_labels_all)
                GT_labels = np.array(GT_labels)
                GT_labels = GT_labels.flatten()      
                correct_num = (Predicted_labels == GT_labels).sum()
                ACC_current = correct_num/GT_labels.shape[0]
                val_ACC[epoch] = ACC_current
                val_loss[epoch] = np.array(loss_current).mean()
                
                if epoch == opt.n_epochs-1:
                    torch.save({'state_dict': net.state_dict(), 'config': opt, 'epoch': epoch}, opt.logdir+'/ckpt/C-Network/Fold_' + str(fold_idx+1) + '_last.pt')        
                
                class_metrics = metrics.class_metrics(Predicted_labels, GT_labels, evidence_)
                AUC_current = class_metrics[0]
                ACC_current = class_metrics[1]
                SEN_current = class_metrics[2]  
                SPE_current = class_metrics[3]
                F1_current = class_metrics[4]
                print('Epoch %d, AUC %.4f, ACC %.4f, SEN %.4f, SPE %.4f, F1 %.4f, Loss %.4f'% (epoch+1, AUC_current, ACC_current, SEN_current, SPE_current, F1_current, val_loss[epoch]))
                if (AUC_best is None) or ((ACC_current >= ACC_best) & (np.sum(class_metrics[2:4]) >= np.sum(metrics_old[2:4])) & (F1_current >= F1_best)):
                    AUC_best = AUC_current
                    ACC_best = ACC_current
                    SEN_best = SEN_current
                    SPE_best = SPE_current
                    F1_best = F1_current
                    iter_best = iter_cnt
                    metrics_old = class_metrics
                    torch.save({'state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'config': opt, 'epoch': epoch}, opt.logdir+'/ckpt/C-Network/Fold_' + str(fold_idx+1) + '_best.pt')
                    print('Save:', opt.logdir+'/ckpt/C-Network/Fold_' + str(fold_idx+1) + '_best.pt')

                print('Current best C-Network iteration %d/%d/%d:'%(iter_best, num_train//opt.batch_size * opt.n_epochs, fold_idx+1), f', AUC: {AUC_best:.2f}', f', ACC: {ACC_best:.2f}', f', SEN: {SEN_best:.2f}', f', SPE: {SPE_best:.2f}', f', F1: {F1_best:.2f}')
                 
                for mod in range(len(opt.modalities)):
                    class_metrics_ = metrics.class_metrics(Predicted_labels_all[:,mod], GT_labels)
                    ACC_current_ = class_metrics_[0]
                    SEN_current_ = class_metrics_[1]  
                    SPE_current_ = class_metrics_[2]
                    F1_current_ = class_metrics_[3]
                    print('modality %d, ACC %.4f, SEN %.4f, SPE %.4f, F1 %.4f'% (mod+1, ACC_current_, SEN_current_, SPE_current_, F1_current_))
                    
                                
            elif opt.train == 'J':                    

                loss_current, loss_current_C, loss_current_G, Predicted_labels, GT_labels, evidence_a = [], [], [], [], [], []
                for j, val_list_ in enumerate(val_groups):
                    dataloader_val = loader_data(opt.dataset,
                                                 opt.data_path, 
                                                 val_list_,
                                                 opt.mod_dict, 
                                                 opt.img_size, 
                                                 batch_size = 1,
                                                 selected_labels = opt.selected_dict,
                                                 renamed_labels = opt.renamed_dict,
                                                 pads = opt.pads,
                                                 num_workers = opt.num_workers,
                                                 shuffle=False)                
                    for i, batch in enumerate(dataloader_val):                        
                        x_input, label_gt = [x.to(device) for x in batch[0]], batch[1].to(device)  
                        with torch.no_grad():
                            x_input, label_gt = check_label(x_input, label_gt)

                            if (x_input[-1].abs().sum() == 0) or (label_gt.shape[0] == 0): 
                                continue   
                             
                            _, x_hat_bottleneck, evidence_acc, x_hat_cls, x_hat_fe_cls, x_hat_fe_cls_s, x_hat_cls_merged, loss_MoC, predicted, uncertainty = net(x_input[-1], epoch, opt.milestone, [1 for _ in range(len(x_input))], label_gt)        
                            gt_cls, gt_cls_merged, gt_bottleneck, gt_fe_cls, gt_fe_cls_s = net.Classifier.ClassifierBlock(x_input)                             
                            
                            J_loss, G_loss, C_loss = criteria_joint(x_input[:-1], x_hat_bottleneck, gt_bottleneck, x_hat_fe_cls, gt_fe_cls, x_hat_fe_cls_s, gt_fe_cls_s, \
                                                                    x_hat_cls, gt_cls, x_hat_cls_merged, gt_cls_merged, loss_MoC, weight_g, weight_c, opt.backbone, device)   
                                
                            loss_current.append(J_loss.cpu())
                            loss_current_C.append(C_loss.cpu())
                            loss_current_G.append(G_loss)
                            Predicted_labels.append(predicted.cpu().numpy())
                            GT_labels.append(label_gt.cpu().numpy())
                            evidence_a.append(evidence_acc[0])
                
                scheduler.step()
                
                Predicted_labels = np.array(Predicted_labels)
                Predicted_labels = Predicted_labels.flatten()
                GT_labels = np.array(GT_labels)
                GT_labels = GT_labels.flatten()      
                correct_num = (Predicted_labels == GT_labels).sum()
                ACC_current = correct_num/GT_labels.shape[0]
                val_ACC[epoch] = ACC_current
                val_loss[epoch] = torch.mean(torch.stack(loss_current)).item()
                val_loss_c[epoch] = torch.mean(torch.stack(loss_current_C)).item()
                    
                if epoch == opt.n_epochs-1:
                    torch.save({'state_dict': net.state_dict(), 'config': opt, 'epoch': epoch}, opt.logdir+'/ckpt/J-Network/Fold_' + str(fold_idx+1) + '_last.pt')
    
                class_metrics = metrics.class_metrics(Predicted_labels, GT_labels, evidence_a)
                AUC_current = class_metrics[0]
                ACC_current = class_metrics[1]
                SEN_current = class_metrics[2]
                SPE_current = class_metrics[3]
                F1_current = class_metrics[4]
                print('Epoch %d, AUC %.2f, ACC %.2f, SEN %.2f, SPE %.2f, F1 %.2f, Loss(all) %.4f, Loss(C) %.4f, Loss(G) %.4f'% (epoch+1, AUC_current, ACC_current, SEN_current, SPE_current, F1_current, val_loss[epoch], val_loss_c[epoch], torch.tensor(loss_current_G).mean()))  
                if (AUC_best is None) or ((ACC_current >= ACC_best) & (np.sum(class_metrics[2:4]) >= np.sum(metrics_old[2:4])) & (F1_current >= F1_best)):
                    AUC_best = AUC_current
                    ACC_best = ACC_current
                    SEN_best = SEN_current
                    SPE_best = SPE_current
                    F1_best = F1_current
                    iter_best = iter_cnt
                    metrics_old = class_metrics
                    torch.save({'state_dict': net.state_dict(), 'config': opt, 'epoch': epoch}, opt.logdir+'/ckpt/J-Network/Fold_' + str(fold_idx+1) + '_best.pt')
                print('Current best J-Network iteration %d/%d/%d:'%(iter_best, num_train//opt.batch_size * opt.n_epochs, fold_idx+1), \
                      f' AUC: {AUC_best:.2f}', f', ACC: {ACC_best:.2f}', f', SEN: {SEN_best:.2f}',f', SPE: {SPE_best:.2f}', f', F1: {F1_best:.2f}')    
                

            if opt.train == 'C':
                np.save(opt.logdir+'/ckpt/C-Network/train_loss_recorded_fold'+str(fold_idx), train_loss) 
                np.save(opt.logdir+'/ckpt/C-Network/val_ACC_recorded_fold'+str(fold_idx), val_ACC)  
            elif opt.train == 'J':
                np.save(opt.logdir+'/ckpt/J-Network/train_loss_recorded_fold'+str(fold_idx), train_loss)  
                np.save(opt.logdir+'/ckpt/J-Network/val_ACC_recorded_fold'+str(fold_idx), val_ACC)        
            
                
         
        ###################  test  ########################
        print('Testing fold %d ...'%(fold_idx+1))
            
        if opt.train == 'C':        
            ckpt = torch.load(opt.logdir+'/ckpt/C-Network/Fold_' + str(fold_idx+1) + '_best.pt')
            cfg = ckpt['config']
            if opt.dataset in ['ADNI', 'OASIS', 'HS']:
                img_size = tuple([112, 128, 112]) 
            elif opt.dataset in ['RJ', 'BraTS']:
                img_size = cfg.img_size

            #### With Uncertainty ####
            if opt.uncertainty == True:                
                net = model.Classifier(opt.dataset, img_size, len(cfg.modalities), cfg.classes, cfg.lambda_epochs, cfg.shared_cls, depth = cfg.stages_gen, chans = cfg.chans_gen, backbone = cfg.backbone).to(device)
            #### Without Uncertainty ####
            else:
                if opt.backbone == '3DCNN':
                    net = model.StandardClassifier(img_size, len(cfg.modalities), cfg.classes, cfg.shared_cls, depth = cfg.stages_gen, chans = cfg.chans_gen).to(device)
                elif opt.backbone == 'PyramidCNN':
                    net = model.StandardMiSePy(opt.dataset, img_size, len(cfg.modalities), cfg.classes, cfg.chans_gen).to(device)
                    
            net.load_state_dict(ckpt['state_dict'])
            net.eval() 
            AUC, ACC, SEN, SPE, F1, Metrics = [], [], [], [], [], []
            for ratio_idx, ratio in enumerate(opt.n_ratio):      
                for std_idx, n_std in enumerate(opt.n_std):
                    for sigma_idx, G_sigma in enumerate(opt.G_sigma):
                        print('Testing...ratiolevel %d, %d/%d'%(ratio, ratio_idx+1, len(opt.n_ratio)))
                        print('Testing...noise level %d, %d/%d'%(n_std, std_idx+1, len(opt.n_std)))
                        print('Testing...blur level %d, %d/%d'%(G_sigma, sigma_idx+1, len(opt.G_sigma)))
                        Predicted_labels, Predicted_labels_all, GT_labels, Uncertainty = [], [], [], []           
                        features_, labels_, views_, evidence_ = [], [], [], []
                        
                        for index_test_data, test_data in enumerate(test_list):            
                            x_input, label_gt = get_testvolume(opt.dataset, opt.data_path, test_data, opt.mod_dict, opt.selected_dict, opt.renamed_dict, pads = opt.pads, n_mean = opt.n_mean, n_std = n_std, crop = opt.img_size, G_kernel = opt.G_kernel, G_sigma = G_sigma, ratio = ratio) 

                            with torch.no_grad():
                                if label_gt == -1e3:     
                                    continue   
                                
                                if opt.uncertainty == True:
                                    #### With Uncertainty ####
                                    evidence, evidence_a, features, loss, predicted, uncertainty, views, _, _ = net(x_input, label_gt, epoch, opt.milestone, [1 for _ in range(len(x_input))])    
                                    _, predicted_all = torch.max(torch.stack(evidence, dim = 1), 2)
                                    Uncertainty.append(uncertainty.cpu().numpy())
                                    features_.append(features)
                                    labels_.append(label_gt.item() * np.ones(len(evidence)))
                                    views_.append(views)
                                    evidence_.append(evidence_a[0])
                                else:
                                    #### Without Uncertainty ####
                                    predicted, loss, output, features = net(x_input, label_gt, 1)                               
                                    features_.append(features)
                                    labels_.append(label_gt.item() * np.ones(len(features)))
                                    predicted_all = predicted.reshape(1,1)
                                    evidence_.append(output)
                                
                                Predicted_labels.append(predicted.cpu().numpy())
                                Predicted_labels_all.append(predicted_all.tolist()[0])
                                GT_labels.append(label_gt.cpu().numpy())
                                
                        torch.save(features_, opt.logdir+'/ckpt/C-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_features.pt')
                        torch.save(labels_, opt.logdir+'/ckpt/C-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_labels.pt')
                        torch.save(views_, opt.logdir+'/ckpt/C-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_views.pt')

                        torch.save(evidence_, opt.logdir+'/ckpt/C-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_evidence.pt')

                        Predicted_labels = np.array(Predicted_labels)
                        Predicted_labels = Predicted_labels.flatten()
                        Predicted_labels_all = np.array(Predicted_labels_all)
                        GT_labels = np.array(GT_labels)
                        GT_labels = GT_labels.flatten()  
                        Uncertainty = np.array(Uncertainty)
                        Uncertainty = Uncertainty.flatten()

                        class_metrics = metrics.class_metrics(Predicted_labels, GT_labels, evidence_)
                        auc = class_metrics[0]
                        acc = class_metrics[1]   
                        sen = class_metrics[2]   
                        spe = class_metrics[3]   
                        f1 = class_metrics[4]   
                        print('====> Fold (best epoch) %d, ACC Repeat: %.4f, AUC Repeat: %.4f, SEN Repeat: %.4f, SPE Repeat: %.4f, F1 Repeat: %.4f'%(fold_idx+1, acc, auc, sen, spe, f1)) 
                                              
                        for mod in range(len(opt.modalities)):
                            class_metrics_ = metrics.class_metrics(Predicted_labels_all[:,mod], GT_labels)
                            ACC_current_ = class_metrics_[0]
                            SEN_current_ = class_metrics_[1]  
                            SPE_current_ = class_metrics_[2]
                            F1_current_ = class_metrics_[3]
                            print('modality %d, ACC %.4f, SEN %.4f, SPE %.4f, F1 %.4f'% (mod+1, ACC_current_, SEN_current_, SPE_current_, F1_current_))

                        Metrics.append(class_metrics)
                      
                        np.save(opt.logdir+'/ckpt/C-Network/Uncertainty_fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx), np.array(Uncertainty))
                        np.save(opt.logdir+'/ckpt/C-Network/PredictedLabel_fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx), np.array(Predicted_labels))
                        np.save(opt.logdir+'/ckpt/C-Network/GTLabel_fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx), np.array(GT_labels))                  

        
              
            np.save(opt.logdir+'/ckpt/C-Network/ACC_fold_'+str(fold_idx+1), np.array(ACC))
            np.save(opt.logdir+'/ckpt/C-Network/AUC_fold_'+str(fold_idx+1), np.array(AUC))
            
              
        elif opt.train == 'J':
            
            ckpt_J = torch.load(opt.logdir+'/ckpt/J-Network/Fold_' + str(fold_idx+1) + '_best.pt')
            cfg = ckpt_J['config']
            if opt.dataset in ['ADNI', 'OASIS', 'HS']:
                img_size = tuple([112, 128, 112])
                #img_size = cfg.img_size
            elif opt.dataset in ['RJ', 'BraTS']:
                img_size = cfg.img_size 
                
            net = model.PreUN(cfg.dataset, img_size, cfg.chans_gen, cfg.stages_gen, len(cfg.modalities), cfg.classes, cfg.lambda_epochs, cfg.shared_cls, cfg.shared_gen, backbone = cfg.backbone).to(device)
            net.load_state_dict(ckpt_J['state_dict'])
            net.eval()            
            AUC, ACC, SPE, SEN, F1, Metrics = [], [], [], [], [], []
            loss_current, loss_current_C, loss_current_G = [], [], []
            for ratio_idx, ratio in enumerate(opt.n_ratio):      
                print('Testing...ratiolevel %d, %d/%d'%(ratio, ratio_idx+1, len(opt.n_ratio)))
                for std_idx, n_std in enumerate(opt.n_std):
                    print('Testing...noise level %d, %d/%d'%(n_std, std_idx+1, len(opt.n_std)))
                    for sigma_idx, G_sigma in enumerate(opt.G_sigma):
                        print('Testing...blur level %d, %d/%d'%(G_sigma, sigma_idx+1, len(opt.G_sigma)))
                        Predicted_labels, Predicted_labels_all, GT_labels, Uncertainty = [], [], [], []
                        features_, labels_, features_trans_, features_gt_, labels_trans_, views_, evidence_a_, features_hat_cls_, features_hat_cls_s_, features_gt_cls_, features_gt_cls_s_ = [], [], [], [], [], [], [], [], [], [], []
            
                        
                        for index_test_data, test_data in enumerate(test_list):            
                            x_input, label_gt = get_testvolume(opt.dataset, opt.data_path, test_data, opt.mod_dict, opt.selected_dict, opt.renamed_dict, opt.pads, n_mean = opt.n_mean, n_std = n_std, crop = opt.img_size, G_kernel = opt.G_kernel, G_sigma = G_sigma, ratio = ratio) 

                            with torch.no_grad():   
                                if label_gt == -1e3: 
                                    continue
                                
                                samples, trans, evidence_acc, x_hat_cls, x_hat_fe_cls, x_hat_fe_cls_s, x_hat_cls_merged, loss_MoC, predicted, uncertainty = net(x_input[-1], epoch, opt.milestone, [1 for _ in range(len(x_input))], label_gt)        
                                _, _, x_gt_fe, _, _, _, views, gt_fe_cls, gt_fe_cls_s = net.Classifier(x_input, label_gt, epoch, opt.milestone, [1 for _ in range(len(x_input))])
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
                                    
                        
                        #### save features for plot ####
                        torch.save(features_, opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_features.pt')
                        torch.save(labels_, opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_labels.pt')
                        torch.save(features_trans_, opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_features_trans.pt')
                        torch.save(labels_trans_, opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_labels_trans.pt')
                        torch.save(features_gt_, opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_features_gt.pt')
                        torch.save(views_, opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_views.pt')
                        torch.save(features_hat_cls_, opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_features_hat_cls.pt')
                        torch.save(features_hat_cls_s_, opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_features_hat_cls_s.pt')
                        torch.save(features_gt_cls_, opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_features_gt_cls.pt')
                        torch.save(features_gt_cls_s_, opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_features_gt_cls_s.pt') 
                        torch.save(evidence_a_, opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx)+'_evidence.pt')
                        

                        Predicted_labels = np.array(Predicted_labels)
                        Predicted_labels = Predicted_labels.flatten()
                        GT_labels = np.array(GT_labels)
                        GT_labels = GT_labels.flatten()   
                        Uncertainty = np.array(Uncertainty)
                        Uncertainty = Uncertainty.flatten()
                        Predicted_labels_all = np.array(Predicted_labels_all)

                        np.save(opt.logdir+'/ckpt/J-Network/Uncertainty_fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx), np.array(Uncertainty))
                        np.save(opt.logdir+'/ckpt/J-Network/PredictedLabel_fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx), np.array(Predicted_labels))
                        np.save(opt.logdir+'/ckpt/J-Network/GTLabel_fold'+str(fold_idx+1)+'_ratio'+str(ratio_idx)+'_std'+str(std_idx)+'_sigma'+str(sigma_idx), np.array(GT_labels))
            

                        class_metrics = metrics.class_metrics(Predicted_labels, GT_labels, evidence_a_)
                        auc = class_metrics[0]
                        acc = class_metrics[1]   
                        sen = class_metrics[2]   
                        spe = class_metrics[3] 
                        f1= class_metrics[4] 
  
                        print('====> Fold (best epoch) %d, ACC Repeat: %.4f, AUC Repeat: %.4f, SEN Repeat: %.4f, SPE Repeat: %.4f, F1 Repeat: %.4f'\
                              %(fold_idx+1, acc, auc, sen, spe, f1)) 

                        for mod in range(len(opt.modalities)):
                            class_metrics_ = metrics.class_metrics(Predicted_labels_all[:,mod], GT_labels)
                            ACC_current_ = class_metrics_[0]
                            SEN_current_ = class_metrics_[1]  
                            SPE_current_ = class_metrics_[2]
                            F1_current_ = class_metrics_[3]
                            print('modality %d, ACC %.4f, SEN %.4f, SPE %.4f, F1 %.4f'% (mod+1, ACC_current_, SEN_current_, SPE_current_, F1_current_))

                        ACC.append(acc)
                        AUC.append(auc)  
                        SPE.append(spe)
                        SEN.append(sen)
                        F1.append(f1) 
                        Metrics.append(class_metrics)

        
            np.save(opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_ACC', np.array(ACC))
            np.save(opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_AUC', np.array(AUC))
            np.save(opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_SPE', np.array(SPE))
            np.save(opt.logdir+'/ckpt/J-Network/fold'+str(fold_idx+1)+'_SEN', np.array(SEN))

        Metrics_folds.append(Metrics)
    
        np.save(opt.logdir+'/ckpt/'+opt.train+'-Network/Metrics_allfolds', np.array(Metrics_folds))

                
            
if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unknown boolean value.')
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=30, \
                        help='number of epochs of training')
    parser.add_argument('--train', type=str, default='C', choices=['C','J'], \
                        help='Model to train in order of Classification->Joint')
    parser.add_argument('--logdir', metavar='logdir', type=str, default='/home/sunkg/Projects/SynthesisPET/models', \
                        help='log directory')
    parser.add_argument('--Load_CNetwork', type=str, default= None, \
                        help='Load the Classification Network')
    parser.add_argument('--Load_JNetwork', type=str, default= None, \
                        help='Load the Joint Network')
    parser.add_argument('--table_csv', type=str, default='/public_bme/data/KaicongSun/SynthesisPET/ADNIALL/ADNI.csv', \
                        help='path to csv file of the scan record')
    parser.add_argument('--data_path', default='/public_bme/data/KaicongSun/SynthesisPET/ADNIALL/ADNI', \
                        type=str, help='path to training data')
    parser.add_argument('--uncertainty', type=str2bool, default=True, \
                        help='use uncertainty, default is true') 
    parser.add_argument('--batch_size', type=int, default=10, \
                        help='size of the batches')
    parser.add_argument('--datablocksize', type=int, default=10, \
                        help='block size of data')
    parser.add_argument('--num_workers', type=int, default=8, \
                        help='number of threads for parallel preprocessing')
    parser.add_argument('--lr', type=float, default=1e-4, \
                        help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.9, \
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, \
                        help='adam: decay of second order momentum of gradient')
    parser.add_argument('--decay_gamma', type=float, default=0.1, \
                        help='adam: decay coefficient of learning rate')
    parser.add_argument('--n_cpu', type=int, default=8, \
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=tuple, default=tuple([112, 128, 112]), \
                        help='size of each image dimension')
    parser.add_argument('--pads', type=tuple, default=tuple([0, 0, 0, 0, 0, 0]), \
                        help='size of each image dimension')
    parser.add_argument('--protocals', metavar='NAME', \
                        type=str, default= ['CN','AD'], nargs='*', help='Groups to classify') 
    parser.add_argument('--backbone', type=str, default='3DCNN', \
                        choices= ['3DCNN','PyramidCNN'], help='Backbone to choose')
    parser.add_argument('--channels', type=int, default=1, \
                        help='number of image channels')
    parser.add_argument('--shared_cls', type=int, default=2, \
                        help='depth of shared layers in classifier')
    parser.add_argument('--shared_gen', type=int, default=0, \
                        help='depth of shared layers in generator')
    parser.add_argument('--modalities', metavar='NAME', \
                        type=str, default= ['FDG', 'AV45', 'MRI'], nargs='*', \
                        help='Modalities to synthesize')
    parser.add_argument('--dataset', type=str, default='ADNI', \
                        choices=['ADNI', 'HS', 'OASIS', 'RJ', 'BraTS'], \
                        help='dataset to choose')
    parser.add_argument('--n_fold', type=int, default=5, \
                        help='XX-fold cross validation') 
    parser.add_argument('--fold_C', type=int, default=None, \
                        help='fold ID of the loaded model C')
    parser.add_argument('--fold_J', type=int, default=None, \
                        help='fold ID of the loaded model PreUN')
    parser.add_argument('--milestone', type=int, default=15, \
                        help='milestone for updating learning rate')           
    parser.add_argument('--lambda_epochs', type=int, default=30, \
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--chans_gen', type=int, default=16, \
                        help='number of channels in Generator')
    parser.add_argument('--stages_gen', type=int, default=4, \
                        help='number of stages in encoder of Generator')
    parser.add_argument('--weight_g', type=tuple, default=tuple([1, 5, 0, 0]), 
                        help='weight of the generator')
    parser.add_argument('--weight_c', type=float, default=5e-4, \
                        help='weight of the classifier')
    parser.add_argument('--n_mean', type=float, default=0, \
                        help='Mean of additive Gaussian noise, default is None') 
    parser.add_argument('--n_std', type=tuple, default=[0], \
                        help='Std of additive Gaussian noise, default is None') 
    parser.add_argument('--n_ratio', type=tuple, default=[0], \
                        help='ratio of data that is contaminated') 
    parser.add_argument('--G_sigma', type=tuple, default=[1e-12], \
                        help='Gaussian blur kernel, default is 1e-12') 
    parser.add_argument('--G_kernel', type=int, default=None, \
                        help='Gaussian kernel size, default is None') 
    parser.add_argument('--n_repeat', type=int, default=1, \
                    help='Repeated time of test data, default is None') 
    parser.add_argument('--save_img', action='store_true', \
                        help='save images or not')
    parser.add_argument('--data_augment', type=str, default= ['None'], nargs='*', \
                        choices=['FDG', 'AV45', 'MRI'], help='add blur and noise to augment training data')
    parser.add_argument('--mod_augment', action='store_true')
    parser.set_defaults(mod_augment=False)

    opt = parser.parse_args()
        
    main(opt)
