import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import ConvBlock, DownConvBlock, DownConvGen, DownConvMiSe


######### evidence-level loss function, referring to paper "Trusted Multi-View Classification with Dynamic Evidential Fusion"
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step, weight):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    label = label.squeeze(1)
    #### weighted loss ####
    weight = torch.FloatTensor([1, weight]).cuda()
    A = torch.sum(weight * label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    #A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    
    #return (A + 0.5*B)
    return (A + B)
       
def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C

    
############# Extractor for auxiliary modalities ################
class Generator(nn.Module):
    def __init__(self, chans, stages, modalities, img_size, classes, lambda_epochs, shared_depth, scale = 16, in_chans = 1):
        super(Generator, self).__init__()

        self.in_chans = in_chans
        self.chans = chans
        self.stages = stages
        self.modalities = modalities
        self.shared_depth = shared_depth
        self.scale = scale

        #### customized layer unshared #### 
        self.encoder_unshared_List = nn.ModuleList([])
        for _ in range(modalities):
            mod_ = []
            mod_.append(ConvBlock(self.in_chans, self.chans))

            chans_ = self.chans * 2**self.shared_depth
            for _ in range(self.stages - self.shared_depth):
                mod_.append(DownConvGen(chans_, chans_*2))
                #mod_.append(DownConvBlock(chans_, chans_*2))
                chans_ *= 2

            mod_.append(ConvBlock(chans_, chans_))
            self.encoder_unshared_List.append(nn.Sequential(*mod_))

        self.reduce_conv = nn.ModuleList([nn.Conv3d(chans_, chans_//self.scale, kernel_size=1) for _ in range(self.modalities)])
        chans_ = chans_//self.scale      
                 
    def encoder_unshared(self, x):
        x_s = []
        for modality in self.encoder_unshared_List:
            x_ = modality(x)
            x_s.append(x_)
        return x_s
                               
    def forward(self, x):
        samples = self.encoder_unshared(x)
        
        for idx in range(len(self.reduce_conv)):
            samples[idx] = self.reduce_conv[idx](samples[idx])

        return samples


################### Classification Block ############################
class ClassifierBlock(nn.Module):
    def __init__(self, img_size, modalities = 3, classes = 3, in_chans = 1, depth = 4, shared_depth = 2, chans = 16, scale = 16):  #scale = 16
        super(ClassifierBlock, self).__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.depth = depth
        self.modalities = modalities
        self.classes = classes
        self.scale = scale
        
        #### customized layers unshared #### 
        self.shared_depth = shared_depth
        self.model_shared = nn.ModuleList([ConvBlock(self.in_chans, chans)])
        for i in range(self.shared_depth):
            self.model_shared.append(DownConvBlock(chans, chans*2))
            chans *= 2         
        self.model_shared = nn.Sequential(*self.model_shared)

        self.model_unshared = nn.ModuleList([])
        for _ in range(modalities):
            mod_ = []
            chans_ = chans
            for _ in range(self.depth - self.shared_depth):
                if chans_ < 256: 
                    mod_.append(DownConvBlock(chans_, chans_*2))
                    chans_ *= 2
                else:
                    mod_.append(DownConvBlock(chans_, chans_))
            self.model_unshared.append(nn.Sequential(*mod_))

        W, H, D = [i//2**self.depth for i in self.img_size]
        self.reduce_conv = nn.ModuleList([nn.Conv3d(chans_, chans_//self.scale, kernel_size=1) for _ in range(self.modalities)])
        chans_ = chans_//self.scale
        
        #### FCN ####            
        self.ind_layers_f = nn.ModuleList([nn.Linear(chans_ * W * H * D, chans_) if self.depth < 5 else nn.Linear(chans_ * 216, chans_) for _ in range(self.modalities)]) # else for stage = 5
      
        self.ind_layers_s = nn.ModuleList([nn.Linear(chans_, self.classes, bias = False) for _ in range(self.modalities-1)])
        self.ind_layers_s += nn.ModuleList([nn.Linear(chans_, self.classes, bias = True)])
        self.FL = nn.Linear(self.modalities * chans_, self.classes, bias = False)
        
        self.activation = nn.Softplus()
        
    def forward(self, imgs, flag = True):
        output = []
        feature_reshaped_s = []
        feature_unshared = []
        feature_reshaped = []
       
        for index in range(len(imgs)):
            out_shared = self.model_shared(imgs[index])
            out_unshared = self.model_unshared[index](out_shared)
            #### reduce dimension ####
            out_unshared = self.reduce_conv[index](out_unshared)
            #### FCN ####
            out_reshaped = out_unshared.view(out_unshared.shape[0], -1)
            out_reshaped = self.ind_layers_f[index](out_reshaped)
                       
            out_reshaped_s = self.ind_layers_s[index](out_reshaped)
            feature_reshaped.append(out_reshaped)
            output.append(self.activation(out_reshaped_s))   
            feature_reshaped_s.append(out_reshaped_s)
            feature_unshared.append(out_unshared)
        
        if flag == False:
            return output, feature_unshared
        else:
            f_merged = torch.concat(feature_reshaped, dim = 1).view(len(imgs[0]), -1)
            output_merged = self.activation(self.FL(f_merged))
                
            return output, output_merged, feature_unshared, feature_reshaped, feature_reshaped_s


        
class MiSePyBlock(nn.Module):
    def __init__(self, dataset, img_size, in_filters = 1, out_filters = 16):
        super(MiSePyBlock, self).__init__()
        if dataset in ['ADNI','HS','OASIS']:
            self.mod_axial_B1 = nn.ModuleList([])
            self.mod_axial_B1.append(DownConvMiSe(in_filters, out_filters, (1, 1, img_size[2]), (1, 1, 1), 0)) # slice-wise conv

            self.mod_axial_B1.append(DownConvMiSe(out_filters, 2*out_filters, (7, 7, 1), (2, 2, 1), 0)) # in_filters, out_filters, ks, stride, pad
            self.mod_axial_B1.append(nn.MaxPool3d((3, 3, 1)))
            self.mod_axial_B1.append(DownConvMiSe(2*out_filters, 4*out_filters, (7, 9, 1), 1, 0))
            self.mod_axial_B1.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B1.append(DownConvMiSe(4*out_filters, 8*out_filters, 1, 1, 0))
            self.seq_axial_B1 = nn.Sequential(*self.mod_axial_B1)

            self.mod_axial_B2 = nn.ModuleList([])
            self.mod_axial_B2.append(DownConvMiSe(in_filters, out_filters, (1, 1, (img_size[2])//2), (1, 1, 1), 0)) # slice-wise conv      
            self.mod_axial_B2.append(DownConvMiSe(out_filters, out_filters, (1, 1, (img_size[2])//2+1), (1, 1, 1), 0)) # slice-wise conv      

            self.mod_axial_B2.append(DownConvMiSe(out_filters, 2*out_filters, (7, 7, 1), (2, 2, 1), 0)) # in_filters, out_filters, ks, stride
            self.mod_axial_B2.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B2.append(DownConvMiSe(2*out_filters, 4*out_filters, (7, 7, 1), 1, 0))
            self.mod_axial_B2.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B2.append(DownConvMiSe(4*out_filters, 8*out_filters, (6, 7, 1), 1, 0))
            self.seq_axial_B2 = nn.Sequential(*self.mod_axial_B2)

            self.mod_axial_B3 = nn.ModuleList([])
            self.mod_axial_B3.append(DownConvMiSe(in_filters, out_filters, (1, 1, (img_size[2])//3+1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_axial_B3.append(DownConvMiSe(out_filters, out_filters, (1, 1, (img_size[2])//3+1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_axial_B3.append(DownConvMiSe(out_filters, out_filters, (1, 1, (img_size[2])//3+1), (1, 1, 1), 0)) # slice-wise conv  

            self.mod_axial_B3.append(DownConvMiSe(out_filters, 2*out_filters, (3, 3, 1), (2, 2, 1), 0)) # in_filters, out_filters, ks, stride
            self.mod_axial_B3.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B3.append(DownConvMiSe(2*out_filters, 4*out_filters, (3, 3, 1), (2, 2, 1), 0))
            self.mod_axial_B3.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B3.append(DownConvMiSe(4*out_filters, 8*out_filters, (2, 2, 1), 1, 0))   
            self.seq_axial_B3 = nn.Sequential(*self.mod_axial_B3)

            #### customized coronal-view #### 
            self.mod_coronal_B1 = nn.ModuleList([])
            self.mod_coronal_B1.append(DownConvMiSe(in_filters, out_filters, (img_size[0], 1, 1), (1, 1, 1), 0)) # slice-wise conv

            self.mod_coronal_B1.append(DownConvMiSe(out_filters, 2*out_filters, (1, 7, 7), (1, 2, 2), 0)) # in_filters, out_filters, ks, stride, pad
            self.mod_coronal_B1.append(nn.MaxPool3d((1, 3, 3)))
            self.mod_coronal_B1.append(DownConvMiSe(2*out_filters, 4*out_filters, (1, 8, 7), 1, 0))
            self.mod_coronal_B1.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B1.append(DownConvMiSe(4*out_filters, 8*out_filters, 1, 1, 0))
            self.seq_coronal_B1 = nn.Sequential(*self.mod_coronal_B1)

            self.mod_coronal_B2 = nn.ModuleList([])
            self.mod_coronal_B2.append(DownConvMiSe(in_filters, out_filters, ((img_size[0])//2, 1, 1), (1, 1, 1), 0)) # slice-wise conv      
            self.mod_coronal_B2.append(DownConvMiSe(out_filters, out_filters, ((img_size[0])//2+1, 1, 1), (1, 1, 1), 0)) # slice-wise conv   

            self.mod_coronal_B2.append(DownConvMiSe(out_filters, 2*out_filters, (1, 7, 7), (1, 2, 2), 0)) # in_filters, out_filters, ks, stride
            self.mod_coronal_B2.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B2.append(DownConvMiSe(2*out_filters, 4*out_filters, (1, 7, 7), 1, 0))
            self.mod_coronal_B2.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B2.append(DownConvMiSe(4*out_filters, 8*out_filters, (1, 7, 6), 1, 0))
            self.seq_coronal_B2 = nn.Sequential(*self.mod_coronal_B2)

            self.mod_coronal_B3 = nn.ModuleList([])
            self.mod_coronal_B3.append(DownConvMiSe(in_filters, out_filters, ((img_size[0])//3+1, 1, 1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_coronal_B3.append(DownConvMiSe(out_filters, out_filters, ((img_size[0])//3+1, 1, 1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_coronal_B3.append(DownConvMiSe(out_filters, out_filters, ((img_size[0])//3+1, 1, 1), (1, 1, 1), 0)) # slice-wise conv  

            self.mod_coronal_B3.append(DownConvMiSe(out_filters, 2*out_filters, (1, 3, 3), (1, 2, 2), 0)) # in_filters, out_filters, ks, stride
            self.mod_coronal_B3.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B3.append(DownConvMiSe(2*out_filters, 4*out_filters, (1, 3, 3), (1, 2, 2), 0))
            self.mod_coronal_B3.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B3.append(DownConvMiSe(4*out_filters, 8*out_filters, (1, 2, 2), 1, 0))
            self.seq_coronal_B3 = nn.Sequential(*self.mod_coronal_B3)

            #### customized sagittal-view ####        
            self.mod_sagittal_B1 = nn.ModuleList([])
            self.mod_sagittal_B1.append(DownConvMiSe(in_filters, out_filters, (1, img_size[1], 1), (1, 1, 1), 0)) # slice-wise conv

            self.mod_sagittal_B1.append(DownConvMiSe(out_filters, 2*out_filters, (7, 1, 7), (2, 1, 2), 0)) # in_filters, out_filters, ks, stride, pad
            self.mod_sagittal_B1.append(nn.MaxPool3d((3, 1, 3)))
            self.mod_sagittal_B1.append(DownConvMiSe(2*out_filters, 4*out_filters, (7, 1, 7), 1, 0))
            self.mod_sagittal_B1.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B1.append(DownConvMiSe(4*out_filters, 8*out_filters, 1, 1, 0))
            self.seq_sagittal_B1 = nn.Sequential(*self.mod_sagittal_B1)

            self.mod_sagittal_B2 = nn.ModuleList([])
            self.mod_sagittal_B2.append(DownConvMiSe(in_filters, out_filters, (1, (img_size[1])//2, 1), (1, 1, 1), 0)) # slice-wise conv      
            self.mod_sagittal_B2.append(DownConvMiSe(out_filters, out_filters, (1, (img_size[1])//2+1, 1), (1, 1, 1), 0)) # slice-wise conv  

            self.mod_sagittal_B2.append(DownConvMiSe(out_filters, 2*out_filters, (7, 1, 7), (2, 1, 2), 0)) # in_filters, out_filters, ks, stride
            self.mod_sagittal_B2.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B2.append(DownConvMiSe(2*out_filters, 4*out_filters, (7, 1, 7), 1, 0))
            self.mod_sagittal_B2.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B2.append(DownConvMiSe(4*out_filters, 8*out_filters, (6, 1, 6), 1, 0))
            self.seq_sagittal_B2 = nn.Sequential(*self.mod_sagittal_B2)

            self.mod_sagittal_B3 = nn.ModuleList([])
            self.mod_sagittal_B3.append(DownConvMiSe(in_filters, out_filters, (1, (img_size[1])//3+1, 1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_sagittal_B3.append(DownConvMiSe(out_filters, out_filters, (1, (img_size[1])//3+1, 1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_sagittal_B3.append(DownConvMiSe(out_filters, out_filters, (1, (img_size[1])//3+2, 1), (1, 1, 1), 0)) # slice-wise conv  

            self.mod_sagittal_B3.append(DownConvMiSe(out_filters, 2*out_filters, (3, 1, 3), (2, 1, 2), 0)) # in_filters, out_filters, ks, stride
            self.mod_sagittal_B3.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B3.append(DownConvMiSe(2*out_filters, 4*out_filters, (3, 1, 3), (2, 1, 2), 0))
            self.mod_sagittal_B3.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B3.append(DownConvMiSe(4*out_filters, 8*out_filters, (2, 1, 2), 1, 0))
            self.seq_sagittal_B3 = nn.Sequential(*self.mod_sagittal_B3)
    
        elif dataset == 'RJ':

            self.mod_axial_B1 = nn.ModuleList([])
            self.mod_axial_B1.append(DownConvMiSe(in_filters, out_filters, (1, 1, img_size[2]), (1, 1, 1), 0)) # slice-wise conv

            self.mod_axial_B1.append(DownConvMiSe(out_filters, 2*out_filters, (7, 7, 1), (2, 2, 1), 0)) # in_filters, out_filters, ks, stride, pad
            self.mod_axial_B1.append(nn.MaxPool3d((3, 3, 1)))
            self.mod_axial_B1.append(DownConvMiSe(2*out_filters, 4*out_filters, (7, 7, 1), 1, 0))
            self.mod_axial_B1.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B1.append(DownConvMiSe(4*out_filters, 8*out_filters, 1, 1, 0))
            self.mod_axial_B1.append(nn.MaxPool3d((2, 2, 1)))
            self.seq_axial_B1 = nn.Sequential(*self.mod_axial_B1)

            self.mod_axial_B2 = nn.ModuleList([])
            self.mod_axial_B2.append(DownConvMiSe(in_filters, out_filters, (1, 1, (img_size[2])//2), (1, 1, 1), 0)) # slice-wise conv      
            self.mod_axial_B2.append(DownConvMiSe(out_filters, out_filters, (1, 1, (img_size[2])//2+1), (1, 1, 1), 0)) # slice-wise conv      

            self.mod_axial_B2.append(DownConvMiSe(out_filters, 2*out_filters, (9, 9, 1), (2, 2, 1), 0)) # in_filters, out_filters, ks, stride
            self.mod_axial_B2.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B2.append(DownConvMiSe(2*out_filters, 4*out_filters, (9, 9, 1), 1, 0))
            self.mod_axial_B2.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B2.append(DownConvMiSe(4*out_filters, 8*out_filters, (7, 9, 1), 1, 0))
            self.mod_axial_B2.append(nn.MaxPool3d((2, 2, 1)))
            self.seq_axial_B2 = nn.Sequential(*self.mod_axial_B2)

            self.mod_axial_B3 = nn.ModuleList([])
            self.mod_axial_B3.append(DownConvMiSe(in_filters, out_filters, (1, 1, (img_size[2])//3+1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_axial_B3.append(DownConvMiSe(out_filters, out_filters, (1, 1, (img_size[2])//3+1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_axial_B3.append(DownConvMiSe(out_filters, out_filters, (1, 1, (img_size[2])//3+2), (1, 1, 1), 0)) # slice-wise conv  

            self.mod_axial_B3.append(DownConvMiSe(out_filters, 2*out_filters, (3, 3, 1), (2, 2, 1), 0)) # in_filters, out_filters, ks, stride
            self.mod_axial_B3.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B3.append(DownConvMiSe(2*out_filters, 4*out_filters, (3, 1, 1), (2, 2, 1), 0))
            self.mod_axial_B3.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B3.append(DownConvMiSe(4*out_filters, 8*out_filters, (1, 1, 1), 1, 0))     
            self.mod_axial_B3.append(nn.MaxPool3d((2, 2, 1)))
            self.seq_axial_B3 = nn.Sequential(*self.mod_axial_B3)

            #### customized coronal-view #### 
            self.mod_coronal_B1 = nn.ModuleList([])
            self.mod_coronal_B1.append(DownConvMiSe(in_filters, out_filters, (img_size[0], 1, 1), (1, 1, 1), 0)) # slice-wise conv

            self.mod_coronal_B1.append(DownConvMiSe(out_filters, 2*out_filters, (1, 7, 7), (1, 2, 2), 0)) # in_filters, out_filters, ks, stride, pad
            self.mod_coronal_B1.append(nn.MaxPool3d((1, 3, 3)))
            self.mod_coronal_B1.append(DownConvMiSe(2*out_filters, 4*out_filters, (1, 8, 7), 1, 0))
            self.mod_coronal_B1.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B1.append(DownConvMiSe(4*out_filters, 8*out_filters, 1, 1, 0))
            self.mod_coronal_B1.append(nn.MaxPool3d((1, 2, 2)))
            self.seq_coronal_B1 = nn.Sequential(*self.mod_coronal_B1)

            self.mod_coronal_B2 = nn.ModuleList([])
            self.mod_coronal_B2.append(DownConvMiSe(in_filters, out_filters, ((img_size[0])//2, 1, 1), (1, 1, 1), 0)) # slice-wise conv      
            self.mod_coronal_B2.append(DownConvMiSe(out_filters, out_filters, ((img_size[0])//2+1, 1, 1), (1, 1, 1), 0)) # slice-wise conv   

            self.mod_coronal_B2.append(DownConvMiSe(out_filters, 2*out_filters, (1, 9, 9), (1, 2, 2), 0)) # in_filters, out_filters, ks, stride
            self.mod_coronal_B2.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B2.append(DownConvMiSe(2*out_filters, 4*out_filters, (1, 9, 9), 1, 0))
            self.mod_coronal_B2.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B2.append(DownConvMiSe(4*out_filters, 8*out_filters, (1, 9, 7), 1, 0))
            self.mod_coronal_B2.append(nn.MaxPool3d((1, 2, 2)))
            self.seq_coronal_B2 = nn.Sequential(*self.mod_coronal_B2)

            self.mod_coronal_B3 = nn.ModuleList([])
            self.mod_coronal_B3.append(DownConvMiSe(in_filters, out_filters, ((img_size[0])//3+1, 1, 1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_coronal_B3.append(DownConvMiSe(out_filters, out_filters, ((img_size[0])//3+1, 1, 1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_coronal_B3.append(DownConvMiSe(out_filters, out_filters, ((img_size[0])//3+2, 1, 1), (1, 1, 1), 0)) # slice-wise conv  

            self.mod_coronal_B3.append(DownConvMiSe(out_filters, 2*out_filters, (1, 3, 3), (1, 2, 2), 0)) # in_filters, out_filters, ks, stride
            self.mod_coronal_B3.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B3.append(DownConvMiSe(2*out_filters, 4*out_filters, (1, 3, 3), (1, 2, 2), 0))
            self.mod_coronal_B3.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B3.append(DownConvMiSe(4*out_filters, 8*out_filters, (1, 1, 1), 1, 0))
            self.mod_coronal_B3.append(nn.MaxPool3d((1, 2, 2)))
            self.seq_coronal_B3 = nn.Sequential(*self.mod_coronal_B3)

            #### customized sagittal-view ####        
            self.mod_sagittal_B1 = nn.ModuleList([])
            self.mod_sagittal_B1.append(DownConvMiSe(in_filters, out_filters, (1, img_size[1], 1), (1, 1, 1), 0)) # slice-wise conv

            self.mod_sagittal_B1.append(DownConvMiSe(out_filters, 2*out_filters, (7, 1, 7), (2, 1, 2), 0)) # in_filters, out_filters, ks, stride, pad
            self.mod_sagittal_B1.append(nn.MaxPool3d((3, 1, 3)))
            self.mod_sagittal_B1.append(DownConvMiSe(2*out_filters, 4*out_filters, (7, 1, 7), 1, 0))
            self.mod_sagittal_B1.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B1.append(DownConvMiSe(4*out_filters, 8*out_filters, 1, 1, 0))
            self.mod_sagittal_B1.append(nn.MaxPool3d((2, 1, 2)))
            self.seq_sagittal_B1 = nn.Sequential(*self.mod_sagittal_B1)

            self.mod_sagittal_B2 = nn.ModuleList([])
            self.mod_sagittal_B2.append(DownConvMiSe(in_filters, out_filters, (1, (img_size[1])//2, 1), (1, 1, 1), 0)) # slice-wise conv      
            self.mod_sagittal_B2.append(DownConvMiSe(out_filters, out_filters, (1, (img_size[1])//2+1, 1), (1, 1, 1), 0)) # slice-wise conv  

            self.mod_sagittal_B2.append(DownConvMiSe(out_filters, 2*out_filters, (9, 1, 9), (2, 1, 2), 0)) # in_filters, out_filters, ks, stride
            self.mod_sagittal_B2.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B2.append(DownConvMiSe(2*out_filters, 4*out_filters, (9, 1, 9), 1, 0))
            self.mod_sagittal_B2.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B2.append(DownConvMiSe(4*out_filters, 8*out_filters, (7, 1, 7), 1, 0))
            self.mod_sagittal_B2.append(nn.MaxPool3d((2, 1, 2)))
            self.seq_sagittal_B2 = nn.Sequential(*self.mod_sagittal_B2)

            self.mod_sagittal_B3 = nn.ModuleList([])
            self.mod_sagittal_B3.append(DownConvMiSe(in_filters, out_filters, (1, (img_size[1])//3+1, 1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_sagittal_B3.append(DownConvMiSe(out_filters, out_filters, (1, (img_size[1])//3+1, 1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_sagittal_B3.append(DownConvMiSe(out_filters, out_filters, (1, (img_size[1])//3+1, 1), (1, 1, 1), 0)) # slice-wise conv  

            self.mod_sagittal_B3.append(DownConvMiSe(out_filters, 2*out_filters, (3, 1, 3), (2, 1, 2), 0)) # in_filters, out_filters, ks, stride
            self.mod_sagittal_B3.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B3.append(DownConvMiSe(2*out_filters, 4*out_filters, (3, 1, 3), (2, 1, 2), 0))
            self.mod_sagittal_B3.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B3.append(DownConvMiSe(4*out_filters, 8*out_filters, (1, 1, 1), 1, 0))
            self.mod_sagittal_B3.append(nn.MaxPool3d((2, 1, 2)))
            self.seq_sagittal_B3 = nn.Sequential(*self.mod_sagittal_B3)

        elif dataset == 'BraTS':
            self.mod_axial_B1 = nn.ModuleList([])
            self.mod_axial_B1.append(DownConvMiSe(in_filters, out_filters, (1, 1, img_size[2]), (1, 1, 1), 0)) # slice-wise conv
    
            self.mod_axial_B1.append(DownConvMiSe(out_filters, 2*out_filters, (9, 9, 1), (2, 2, 1), 0)) # in_filters, out_filters, ks, stride, pad
            self.mod_axial_B1.append(nn.MaxPool3d((3, 3, 1)))
            self.mod_axial_B1.append(DownConvMiSe(2*out_filters, 4*out_filters, (8, 8, 1), 1, 0))
            self.mod_axial_B1.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B1.append(DownConvMiSe(4*out_filters, 8*out_filters, 1, 1, 0))
            self.mod_axial_B1.append(nn.MaxPool3d((2, 2, 1)))
            self.seq_axial_B1 = nn.Sequential(*self.mod_axial_B1)
    
            self.mod_axial_B2 = nn.ModuleList([])
            self.mod_axial_B2.append(DownConvMiSe(in_filters, out_filters, (1, 1, (img_size[2])//2), (1, 1, 1), 0)) # slice-wise conv      
            self.mod_axial_B2.append(DownConvMiSe(out_filters, out_filters, (1, 1, (img_size[2])//2+1), (1, 1, 1), 0)) # slice-wise conv      
    
            self.mod_axial_B2.append(DownConvMiSe(out_filters, 2*out_filters, (9, 9, 1), (2, 2, 1), 0)) # in_filters, out_filters, ks, stride
            self.mod_axial_B2.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B2.append(DownConvMiSe(2*out_filters, 4*out_filters, (9, 9, 1), 1, 0))
            self.mod_axial_B2.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B2.append(DownConvMiSe(4*out_filters, 8*out_filters, (9, 9, 1), 1, 0))
            self.mod_axial_B2.append(nn.MaxPool3d((2, 2, 1)))
            self.seq_axial_B2 = nn.Sequential(*self.mod_axial_B2)
    
            self.mod_axial_B3 = nn.ModuleList([])
            self.mod_axial_B3.append(DownConvMiSe(in_filters, out_filters, (1, 1, (img_size[2])//3+1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_axial_B3.append(DownConvMiSe(out_filters, out_filters, (1, 1, (img_size[2])//3+1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_axial_B3.append(DownConvMiSe(out_filters, out_filters, (1, 1, (img_size[2])//3), (1, 1, 1), 0)) # slice-wise conv  
    
            self.mod_axial_B3.append(DownConvMiSe(out_filters, 2*out_filters, (3, 3, 1), (2, 2, 1), 0)) # in_filters, out_filters, ks, stride
            self.mod_axial_B3.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B3.append(DownConvMiSe(2*out_filters, 4*out_filters, (3, 3, 1), (2, 2, 1), 0))
            self.mod_axial_B3.append(nn.MaxPool3d((2, 2, 1)))
            self.mod_axial_B3.append(DownConvMiSe(4*out_filters, 8*out_filters, (2, 2, 1), 1, 0))     
            self.mod_axial_B3.append(nn.MaxPool3d((2, 2, 1)))
            self.seq_axial_B3 = nn.Sequential(*self.mod_axial_B3)
    
            #### customized coronal-view #### 
            self.mod_coronal_B1 = nn.ModuleList([])
            self.mod_coronal_B1.append(DownConvMiSe(in_filters, out_filters, (img_size[0], 1, 1), (1, 1, 1), 0)) # slice-wise conv
    
            self.mod_coronal_B1.append(DownConvMiSe(out_filters, 2*out_filters, (1, 9, 9), (1, 2, 2), 0)) # in_filters, out_filters, ks, stride, pad
            self.mod_coronal_B1.append(nn.MaxPool3d((1, 3, 3)))
            self.mod_coronal_B1.append(DownConvMiSe(2*out_filters, 4*out_filters, (1, 9, 9), 1, 0))
            self.mod_coronal_B1.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B1.append(DownConvMiSe(4*out_filters, 8*out_filters, 1, 1, 0))
            self.mod_coronal_B1.append(nn.MaxPool3d((1, 2, 2)))
            self.seq_coronal_B1 = nn.Sequential(*self.mod_coronal_B1)
    
            self.mod_coronal_B2 = nn.ModuleList([])
            self.mod_coronal_B2.append(DownConvMiSe(in_filters, out_filters, ((img_size[0])//2, 1, 1), (1, 1, 1), 0)) # slice-wise conv      
            self.mod_coronal_B2.append(DownConvMiSe(out_filters, out_filters, ((img_size[0])//2+1, 1, 1), (1, 1, 1), 0)) # slice-wise conv   
    
            self.mod_coronal_B2.append(DownConvMiSe(out_filters, 2*out_filters, (1, 10, 10), (1, 2, 2), 0)) # in_filters, out_filters, ks, stride
            self.mod_coronal_B2.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B2.append(DownConvMiSe(2*out_filters, 4*out_filters, (1, 9, 9), 1, 0))
            self.mod_coronal_B2.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B2.append(DownConvMiSe(4*out_filters, 8*out_filters, (1, 9, 7), 1, 0))
            self.mod_coronal_B2.append(nn.MaxPool3d((1, 2, 2)))
            self.seq_coronal_B2 = nn.Sequential(*self.mod_coronal_B2)
    
            self.mod_coronal_B3 = nn.ModuleList([])
            self.mod_coronal_B3.append(DownConvMiSe(in_filters, out_filters, ((img_size[0])//3+1, 1, 1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_coronal_B3.append(DownConvMiSe(out_filters, out_filters, ((img_size[0])//3+1, 1, 1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_coronal_B3.append(DownConvMiSe(out_filters, out_filters, ((img_size[0])//3, 1, 1), (1, 1, 1), 0)) # slice-wise conv  
    
            self.mod_coronal_B3.append(DownConvMiSe(out_filters, 2*out_filters, (1, 3, 3), (1, 2, 2), 0)) # in_filters, out_filters, ks, stride
            self.mod_coronal_B3.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B3.append(DownConvMiSe(2*out_filters, 4*out_filters, (1, 3, 3), (1, 2, 2), 0))
            self.mod_coronal_B3.append(nn.MaxPool3d((1, 2, 2)))
            self.mod_coronal_B3.append(DownConvMiSe(4*out_filters, 8*out_filters, (1, 2, 2), 1, 0))
            self.mod_coronal_B3.append(nn.MaxPool3d((1, 2, 2)))
            self.seq_coronal_B3 = nn.Sequential(*self.mod_coronal_B3)
    
            #### customized sagittal-view ####        
            self.mod_sagittal_B1 = nn.ModuleList([])
            self.mod_sagittal_B1.append(DownConvMiSe(in_filters, out_filters, (1, img_size[1], 1), (1, 1, 1), 0)) # slice-wise conv
    
            self.mod_sagittal_B1.append(DownConvMiSe(out_filters, 2*out_filters, (9, 1, 9), (2, 1, 2), 0)) # in_filters, out_filters, ks, stride, pad
            self.mod_sagittal_B1.append(nn.MaxPool3d((3, 1, 3)))
            self.mod_sagittal_B1.append(DownConvMiSe(2*out_filters, 4*out_filters, (9, 1, 9), 1, 0))
            self.mod_sagittal_B1.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B1.append(DownConvMiSe(4*out_filters, 8*out_filters, 1, 1, 0))
            self.mod_sagittal_B1.append(nn.MaxPool3d((2, 1, 2)))
            self.seq_sagittal_B1 = nn.Sequential(*self.mod_sagittal_B1)
    
            self.mod_sagittal_B2 = nn.ModuleList([])
            self.mod_sagittal_B2.append(DownConvMiSe(in_filters, out_filters, (1, (img_size[1])//2, 1), (1, 1, 1), 0)) # slice-wise conv      
            self.mod_sagittal_B2.append(DownConvMiSe(out_filters, out_filters, (1, (img_size[1])//2+1, 1), (1, 1, 1), 0)) # slice-wise conv  
    
            self.mod_sagittal_B2.append(DownConvMiSe(out_filters, 2*out_filters, (9, 1, 9), (2, 1, 2), 0)) # in_filters, out_filters, ks, stride
            self.mod_sagittal_B2.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B2.append(DownConvMiSe(2*out_filters, 4*out_filters, (9, 1, 9), 1, 0))
            self.mod_sagittal_B2.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B2.append(DownConvMiSe(4*out_filters, 8*out_filters, (9, 1, 7), 1, 0))
            self.mod_sagittal_B2.append(nn.MaxPool3d((2, 1, 2)))
            self.seq_sagittal_B2 = nn.Sequential(*self.mod_sagittal_B2)
    
            self.mod_sagittal_B3 = nn.ModuleList([])
            self.mod_sagittal_B3.append(DownConvMiSe(in_filters, out_filters, (1, (img_size[1])//3+1, 1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_sagittal_B3.append(DownConvMiSe(out_filters, out_filters, (1, (img_size[1])//3+1, 1), (1, 1, 1), 0)) # slice-wise conv  
            self.mod_sagittal_B3.append(DownConvMiSe(out_filters, out_filters, (1, (img_size[1])//3, 1), (1, 1, 1), 0)) # slice-wise conv  
    
            self.mod_sagittal_B3.append(DownConvMiSe(out_filters, 2*out_filters, (3, 1, 3), (2, 1, 2), 0)) # in_filters, out_filters, ks, stride
            self.mod_sagittal_B3.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B3.append(DownConvMiSe(2*out_filters, 4*out_filters, (3, 1, 3), (2, 1, 2), 0))
            self.mod_sagittal_B3.append(nn.MaxPool3d((2, 1, 2)))
            self.mod_sagittal_B3.append(DownConvMiSe(4*out_filters, 8*out_filters, (2, 1, 2), 1, 0))
            self.mod_sagittal_B3.append(nn.MaxPool3d((2, 1, 2)))
            self.seq_sagittal_B3 = nn.Sequential(*self.mod_sagittal_B3)

    def forward(self, imgs):
        A1 = self.seq_axial_B1(imgs)
        A2 = self.seq_axial_B2(imgs)
        A3 = self.seq_axial_B3(imgs)

        C1 = self.seq_coronal_B1(imgs)
        C2 = self.seq_coronal_B2(imgs)
        C3 = self.seq_coronal_B3(imgs)

        S1 = self.seq_sagittal_B1(imgs)
        S2 = self.seq_sagittal_B2(imgs)
        S3 = self.seq_sagittal_B3(imgs)

        A = A1 + A2 + A3
        S = S1 + S2 + S3
        C = C1 + C2 + C3

        output = torch.concat([A.view(A.shape[0],-1), S.view(S.shape[0],-1), C.view(C.shape[0],-1)], dim = 1) # B, C*3
        return output


class GeneratorMiSePy(nn.Module):
    def __init__(self, dataset, chans, modalities, img_size, classes, scale = 8):
        super(GeneratorMiSePy, self).__init__()

        self.modalities = modalities
        self.scale = scale
        self.img_size = img_size

        #### customized layer unshared #### 
        self.encoder_unshared_List = nn.ModuleList([])
        for _ in range(modalities):
            miSePyBlock = MiSePyBlock(dataset, img_size, out_filters = chans)
            self.encoder_unshared_List.append(miSePyBlock)
        chans_ = chans * 8
        
        if dataset == 'BraTS':
            fs = 5*5 + 5*3 + 5*3
        elif dataset in ['ADNI', 'HS', 'OASIS', 'RJ']:
            fs = 5*6 + 5*6 + 5*5
        
        self.reduce_conv = nn.ModuleList([nn.Linear(fs*chans_, chans_) for _ in range(self.modalities)])
                         
    def encoder_unshared(self, x):
        x_s = []
        for modality in self.encoder_unshared_List:
            x_ = modality(x)
            x_s.append(x_)
        return x_s
                              
    
    def forward(self, x):
        samples = self.encoder_unshared(x)
        
        for idx in range(len(self.reduce_conv)):
            samples[idx] = self.reduce_conv[idx](samples[idx])

        return samples


class ClassifierMiSePyBlock(nn.Module):
    def __init__(self, dataset, img_size, modalities = 3, classes = 2, chans = 8, scale = 8):
        super(ClassifierMiSePyBlock, self).__init__()
        self.img_size = img_size
        self.modalities = modalities
        self.classes = classes
        self.scale = scale
        
        #### customized layers unshared #### 
        self.model_unshared = nn.ModuleList([])
        for _ in range(modalities):
            miSePyBlock = MiSePyBlock(dataset, img_size, out_filters = chans)
            self.model_unshared.append(miSePyBlock)

        chans_ = chans * 8

        if dataset == 'BraTS':
            fs = 5*5 + 5*3 + 5*3
        elif dataset in ['ADNI', 'HS', 'OASIS', 'RJ']:
            fs = 5*6 + 5*6 + 5*5

        self.reduce_conv = nn.ModuleList([nn.Linear(fs*chans_, chans_) for _ in range(self.modalities)])
        
        #### FCN ####            
        self.ind_layers_f = nn.ModuleList([nn.Linear(chans_, chans_//self.scale) for _ in range(self.modalities)])
        chans_ = chans_//self.scale
        #### Pooling ####
        self.ind_layers_s = nn.ModuleList([nn.Linear(chans_, self.classes, bias = True) for _ in range(self.modalities)])
        self.FL = nn.Linear(self.modalities * chans_, self.classes, bias = True)       
        
        self.activation = nn.Softplus()
        
    def forward(self, imgs, flag = True):
        output = []
        feature_reshaped_s = []
        feature_unshared = []
        feature_reshaped = []
        for index in range(len(imgs)):
            out_unshared = self.model_unshared[index](imgs[index])
            #### reduce dimension ####
            out_unshared = self.reduce_conv[index](out_unshared)
            #### FCN ####
            out_reshaped = out_unshared.view(out_unshared.shape[0], -1)

            out_reshaped = self.ind_layers_f[index](out_reshaped)

            out_reshaped_s = self.ind_layers_s[index](out_reshaped)
            feature_reshaped.append(out_reshaped)
            output.append(self.activation(out_reshaped_s))   
            feature_reshaped_s.append(out_reshaped_s)
            feature_unshared.append(out_unshared)
        
        if flag == False:
            return output, feature_unshared
        else:
            f_merged = torch.concat(feature_reshaped, dim = 1).view(len(imgs[0]), -1)
            output_merged = self.activation(self.FL(f_merged))
                
            return output, output_merged, feature_unshared, feature_reshaped, feature_reshaped_s


##################### Multi-modal Classification model with uncertainty estimation used in Stage I ########################    
class Classifier(nn.Module):

    def __init__(self, dataset, img_size, modalities, classes, lambda_epochs = 1, shared_depth = 2, in_chans = 1, depth_sc = 1, depth = 4, chans = 16, backbone = '3DCNN'):
        """
        :param classes: Number of classification categories
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Classifier, self).__init__()
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.depth = depth
        self.chans = chans
        if backbone == '3DCNN':
            self.ClassifierBlock = ClassifierBlock(img_size, modalities, classes, in_chans, depth, shared_depth, chans)
        elif backbone == 'PyramidCNN':
            self.ClassifierBlock = ClassifierMiSePyBlock(dataset, img_size, modalities, classes, chans)


    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a

            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a, u_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a, u_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a, u_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a, u_a


    def forward(self, X, y, global_step, milestone, weight):
        evidence, merged_evidence, fe_unshared, fe_cls, fe_cls_s = self.ClassifierBlock(X)
        loss = torch.zeros(1, 1).cuda()
        alpha_a = []
        u_a = []
        evidence_a = []
        views = []
        for idx in range(len(X[-1])):
            alpha_v = []
            view = []
            
            for v_num in range(len(X)):
                if (X[v_num][idx].abs().sum() != 0) & (weight[v_num]>=1/2) & (weight[v_num]<=2):  # 2 is the threshold for data imbalance, should be the same scalar as used in the train.py
                    view.append(True)
                    #### individual evidence ####
                    alpha_v.append(evidence[v_num][idx].unsqueeze(0) + 1)         
                    loss += ce_loss(y[idx].unsqueeze(0), alpha_v[-1], self.classes, global_step, self.lambda_epochs, weight[v_num])                
                   
                else:
                    view.append(False)
           
            #### intergrate merged evidence #### 
            alpha_v.append(merged_evidence[idx].unsqueeze(0) + 1)         
            loss += ce_loss(y[idx].unsqueeze(0), alpha_v[-1], self.classes, global_step, self.lambda_epochs, weight[-1])    
            
            views.append(view)  
            
            if len(alpha_v) == 1:
                alpha_a.append(alpha_v[0])
                u_a.append(self.classes/torch.sum(alpha_a[-1], axis=1).unsqueeze(0))
            elif len(alpha_v) > 1:
                tmp1, tmp2 = self.DS_Combin(alpha_v)
                alpha_a.append(tmp1)
                u_a.append(tmp2)
            
            if len(alpha_v) >= 1:
                evidence_a.append(alpha_a[-1] - 1)
                loss += ce_loss(y[idx].unsqueeze(0), alpha_a[-1], self.classes, global_step, self.lambda_epochs, weight[-1])
            
        loss = torch.mean(loss)
        
        out, predicted = torch.max(torch.stack(evidence_a, dim = 1), 2)
        return evidence, evidence_a, fe_unshared, loss, predicted, torch.stack(u_a, dim = 1), views, fe_cls, fe_cls_s



##################### Multi-modal Classification model with uncertainty estimation used in Stage II ########################    
class PreUN(nn.Module):
    def __init__(self, dataset, img_size, chans_gen, stages_gen, modalities, classes, lambda_epochs, shared_cls, shared_gen, backbone):
        super(PreUN, self).__init__()
        self.Classifier = Classifier(dataset, img_size, modalities, classes, lambda_epochs, shared_cls, depth = stages_gen, chans = chans_gen, backbone = backbone)
        self.backbone = backbone
        if backbone =='3DCNN':
            self.Generator = Generator(chans_gen, stages_gen, modalities-1, img_size, classes, lambda_epochs, shared_gen)
        elif backbone == 'PyramidCNN':
            self.Generator = GeneratorMiSePy(dataset, chans_gen, modalities-1, img_size, classes)
        self.lambda_epochs = lambda_epochs
        self.modalities = modalities
            
        self.classes = classes
        self.activation = nn.Softplus()
        
        
    def forward(self, imgs, epoch_index, milestone, weight, labels = None):
        codes = self.Generator(imgs)
        epoch_index = epoch_index + 1

        evidence = []
        out_trans = []
        out_f = []
        out_s = []
                           
        if self.backbone == '3DCNN':
            out_shared_ = self.Classifier.ClassifierBlock.model_shared(imgs)
            out_unshared_ = self.Classifier.ClassifierBlock.model_unshared[-1](out_shared_)
        elif self.backbone == 'PyramidCNN':
            out_unshared_ = self.Classifier.ClassifierBlock.model_unshared[-1](imgs)

        out_unshared_ = self.Classifier.ClassifierBlock.reduce_conv[-1](out_unshared_)
        codes.append(out_unshared_)
        
        for idx in range(self.modalities):
            out_reshaped_ = codes[idx].view(codes[idx].shape[0], -1)   
            out_reshaped_f = self.Classifier.ClassifierBlock.ind_layers_f[idx](out_reshaped_) # use sigmoid as indicator
            out_reshaped_s = self.Classifier.ClassifierBlock.ind_layers_s[idx](out_reshaped_f)
            evidence.append(self.activation(out_reshaped_s))  
            out_trans.append(out_reshaped_)
            out_f.append(out_reshaped_f)          
            out_s.append(out_reshaped_s)

        f_merged = torch.concat(out_f, dim = 1).view(len(codes[-1]), -1)
        output_merged = self.activation(self.Classifier.ClassifierBlock.FL(f_merged))
        
        #### Dirichlet ####
        loss = 0
        alpha_a = []
        u_a = []
        evidence_a = []
        for idx in range(len(evidence[-1])):
            alpha_v = []
            if epoch_index-1 < milestone: # train the pseudo branches
                for v_num in range(len(evidence)-1):
                    if (weight[v_num]>=1/5) & (weight[v_num]<=5): # 5 is the threshold for data imbalance, should be the same scalar as used in the train.py
                        #### individual evidence ####
                        alpha_v.append(evidence[v_num][idx].unsqueeze(0) + 1)         
                        loss += ce_loss(labels[idx].unsqueeze(0), alpha_v[-1], self.classes, epoch_index, self.lambda_epochs, weight[v_num])     

            else: # train all the branches
                for v_num in range(len(evidence)):
                    if (weight[v_num]>=1/5) & (weight[v_num]<=5): # 5 is the threshold for data imbalance
                        #### individual evidence ####
                        alpha_v.append(evidence[v_num][idx].unsqueeze(0) + 1)         
                        loss += ce_loss(labels[idx].unsqueeze(0), alpha_v[-1], self.classes, epoch_index, self.lambda_epochs, weight[v_num])     
    
                #### add merged evidence #### 
                alpha_v.append(output_merged[idx].unsqueeze(0) + 1)      
                loss += ce_loss(labels[idx].unsqueeze(0), alpha_v[-1], self.classes, epoch_index, self.lambda_epochs, weight[-1])    
                
                    
            if len(alpha_v) == 1:
                alpha_a.append(alpha_v[0])
                u_a.append(self.classes/torch.sum(alpha_a[-1], axis=1).unsqueeze(0))
            else:
                tmp1, tmp2 = self.Classifier.DS_Combin(alpha_v)
                alpha_a.append(tmp1)
                u_a.append(tmp2)
                
            
            evidence_a.append(alpha_a[-1] - 1) 
            loss += ce_loss(labels[idx].unsqueeze(0), alpha_a[-1], self.classes, epoch_index, self.lambda_epochs, weight[-1])
            
        loss = torch.mean(loss)
  
        _, predicted = torch.max(torch.stack(evidence_a, dim = 1), 2)     
        
        return codes, out_trans, evidence_a, evidence, out_f, out_s, output_merged, loss, predicted, torch.stack(u_a, dim = 1)



################### Classification model without uncertainty estimation (3D CNN Backbone) ############################
class StandardClassifier(nn.Module):
    def __init__(self, img_size, modalities = 3, classes = 3, shared_depth = 2, in_chans = 1, depth = 4, chans = 16, scale = 16):
        super(StandardClassifier, self).__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.depth = depth
        self.modalities = modalities
        self.classes = classes
        self.scale = scale

        #### customized layers unshared #### 
        self.shared_depth = shared_depth
        self.model_shared = nn.ModuleList([ConvBlock(self.in_chans, chans)])
        for i in range(self.shared_depth):
            self.model_shared.append(DownConvBlock(chans, chans*2))
            chans *= 2         
        self.model_shared = nn.Sequential(*self.model_shared)

        self.model_unshared = nn.ModuleList([])
        for _ in range(modalities):
            mod_ = []
            chans_ = chans
            for _ in range(self.depth - self.shared_depth):
                if chans_ < 256: 
                    mod_.append(DownConvBlock(chans_, chans_*2))
                    chans_ *= 2
                else:
                    mod_.append(DownConvBlock(chans_, chans_))
            self.model_unshared.append(nn.Sequential(*mod_))

        W, H, D = [i//2**self.depth for i in self.img_size]

        #### reduce dimension ####
        self.reduce_conv = nn.ModuleList([nn.Conv3d(chans_, chans_//self.scale, kernel_size=1) for _ in range(self.modalities)])
        chans_ = chans_//self.scale
        #### fully connected ####
        self.adv_layers = nn.ModuleList([nn.Linear(chans_ * W * H * D, chans_, bias = True) if self.depth < 5 else nn.Linear(chans_ * 216, chans_, bias = True) for _ in range(self.modalities)])
        self.FL = nn.Linear(self.modalities * chans_, self.classes, bias = False)
        
        self.activation = nn.Softmax()

    def forward(self, imgs, labels, weight):
        output = []
        feature_unshared = []
        Loss = torch.nn.CrossEntropyLoss(torch.FloatTensor([1, weight]).cuda())
        for index in range(len(imgs)):
            out_shared = self.model_shared(imgs[index])
            out_unshared = self.model_unshared[index](out_shared)
            #### reduce dimension ####
            out_unshared = self.reduce_conv[index](out_unshared)
            #### fully connected ####
            out_reshaped = out_unshared.view(out_unshared.shape[0], -1)
            out_reshaped = self.adv_layers[index](out_reshaped)
            feature_unshared.append(out_reshaped)
                    
        f_merged = torch.concat(feature_unshared, dim = 1).view(len(imgs[0]), -1)
        output = self.activation(self.FL(f_merged))
        _, predicted = torch.max(output, 1)
        loss = Loss(output, F.one_hot(torch.swapaxes(labels, 0, 1), num_classes=self.classes).squeeze(0).float())
        return predicted, loss, output, feature_unshared
    
    
################### Classification model without uncertainty estimation (MiSePyNet Backbone) ############################
class StandardMiSePy(nn.Module):
    def __init__(self, dataset, img_size, modalities = 3, classes = 3, chans = 8, scale = 8):
        super(StandardMiSePy, self).__init__()
        self.img_size = img_size
        self.modalities = modalities
        self.classes = classes
        self.scale = scale
        
        #### customized layers unshared #### 
        self.model_unshared = nn.ModuleList([])
        for _ in range(modalities):
            miSePyBlock = MiSePyBlock(dataset, img_size, out_filters = chans)
            self.model_unshared.append(miSePyBlock)

        chans_ = chans * 8
        
        if dataset == 'BraTS':
            fs = 5*5 + 5*3 + 5*3
        elif dataset in ['ADNI', 'HS', 'OASIS', 'RJ']:
            fs = 5*6 + 5*6 + 5*5

        self.reduce_conv = nn.ModuleList([nn.Linear(fs*chans_, chans_) for _ in range(self.modalities)])
        
        #### FCN ####            
        self.ind_layers_f = nn.ModuleList([nn.Linear(chans_, chans_//self.scale) for _ in range(self.modalities)])
        chans_ = chans_//self.scale
        self.FL = nn.Linear(self.modalities * chans_, self.classes, bias = True)
        
        self.activation = nn.ReLU()
        
    def forward(self, imgs, labels, weight):
        output = []
        feature_unshared = []
        Loss = torch.nn.CrossEntropyLoss(torch.FloatTensor([1, weight]).cuda())
        for index in range(len(imgs)):
            out_unshared = self.model_unshared[index](imgs[index])
            
            #### reduce dimension ####
            out_unshared = self.reduce_conv[index](out_unshared)
            #### FCN ####
            out_reshaped = out_unshared.view(out_unshared.shape[0], -1)
            out_reshaped = self.ind_layers_f[index](out_reshaped)

            feature_unshared.append(out_reshaped)
        
        f_merged = torch.concat(feature_unshared, dim = 1).view(len(imgs[0]), -1)            
        output = self.activation(self.FL(f_merged))
        _, predicted = torch.max(output, 1)

        loss = Loss(output, F.one_hot(torch.swapaxes(labels, 0, 1), num_classes=self.classes).squeeze(0).float())
        return predicted, loss, output, feature_unshared
