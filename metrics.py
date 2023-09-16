import numpy as np
import skimage
try:
    import skimage.metrics
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
except ImportError:
    import skimage.measure
    from skimage.measure import compare_psnr, compare_ssim
from sklearn import metrics
import torch.nn.functional as F
import torch
import utils
import math 


def to_numpy(*args):
    outputs = []
    for arg in args:
        if hasattr(arg, 'cpu') and callable(arg.cpu):
            arg = arg.detach().cpu()
        if hasattr(arg, 'numpy') and callable(arg.numpy):
            arg = arg.detach().numpy()
        outputs.append(arg)
    return outputs


def mse(gt, pred, gt_flag=True):
    if gt_flag==True:
        gt, pred = to_numpy(gt, pred)
        mse = np.mean((gt - pred) ** 2).item()
    else:
        mse = -1000
    return mse


def mae(gt, pred, gt_flag=True):
    if gt_flag==True:
        gt, pred = to_numpy(gt, pred)
        mae = np.mean(np.absolute(gt - pred)).item()
    else:
        mae = -1000
    return mae


def nmse(gt, pred, gt_flag=True):
    if gt_flag==True:
        gt, pred = to_numpy(gt, pred)
        nmse = (np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2).item()
    else:
        nmse = -1000
    return nmse


def psnr(gt, pred, gt_flag=True):
    if gt_flag==True:
        gt, pred = to_numpy(gt, pred)
        gt = (gt-gt.min())/(gt.max()-gt.min())
        pred = (pred-pred.min())/(pred.max()-pred.min())
        psnr = compare_psnr(gt, pred).item()
    else:
        psnr = -1000
    return psnr


def ssim(gt, pred, gt_flag=True):
    if gt_flag==True:
        gt, pred = to_numpy(gt, pred)
        gt = (gt-gt.min())/(gt.max()-gt.min())
        pred = (pred-pred.min())/(pred.max()-pred.min())
        ssim = np.mean([compare_ssim(g[0], p[0]) \
                for g, p in zip(gt, pred)]).item()
    else:
        ssim = -1000
    return ssim


def dice(gt, pred, label=None):
    gt, pred = to_numpy(gt, pred)
    if label is None:
        gt, pred = gt.astype(np.bool), pred.astype(np.bool)
    else:
        gt, pred = (gt == label), (pred == label)
    intersection = np.logical_and(gt, pred)
    return 2.*intersection.sum() / (gt.sum() + pred.sum())


from scipy.special import xlogy
def mi(gt, pred, bins=64, minVal=0, maxVal=1):
    assert gt.shape == pred.shape
    gt, pred = to_numpy(gt, pred)
    mi = []
    for x, y in zip(gt, pred):
        Pxy = np.histogram2d(x.ravel(), y.ravel(), bins, \
                range=((minVal,maxVal),(minVal,maxVal)))[0]
        Pxy = Pxy/(Pxy.sum()+1e-10)
        Px = Pxy.sum(axis=1)
        Py = Pxy.sum(axis=0)
        PxPy = Px[..., None]*Py[None, ...]
        result = xlogy(Pxy, Pxy) - xlogy(Pxy, PxPy)
        mi.append(result.sum())
    return np.mean(mi).item()


def lncc_loss(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

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

    return -1 * torch.mean(cc)


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


def ms_lncc_loss(I, J, win=None, ms=3, sigma=3):
    smooth_fn = lambda x: torch.nn.functional.avg_pool2d( \
            utils.gaussian_smooth(x, sigma), kernel_size = 2, stride=2)
    loss = lncc_loss(I, J, win)
    for _ in range(ms - 1):
        I, J = map(smooth_fn, (I, J))
        loss = loss + lncc_loss(I, J, win)
    return loss / ms


def synthesis_metrics(x_hats, x_gts):
    PSNR, SSIM = [], []
    for x_hat, x_gt in zip(x_hats, x_gts):
        #### ignore the data without MRI input ####
        if x_gt.sum() == 0: 
            continue  
        psnr_ = psnr(x_gt, x_hat)
        ssim_ = ssim(x_gt, x_hat)
        PSNR.append(psnr_)
        SSIM.append(ssim_)
    PSNR = np.array(PSNR or 0).mean()
    SSIM = np.array(SSIM or 0).mean()
    return PSNR, SSIM


def class_metrics(testvals, labels, probs = None):
    if probs != None:
        prods_ = [i[0][1].cpu().detach().numpy() for i in probs]
        AUC = metrics.roc_auc_score(y_score=np.transpose(prods_), y_true=np.transpose(labels), average='samples')
    
    TP = 0; TN=0; FP=0; FN=0
    for idx in range(len(testvals)):
        if (labels[idx]==1) & (testvals[idx]==1):
            TP = TP + 1
        elif (labels[idx]==0) & (testvals[idx]==0):
            TN = TN + 1
        elif (labels[idx]==0) & (testvals[idx]==1):
            FP = FP + 1
        elif (labels[idx]==1) & (testvals[idx]==0):
            FN = FN + 1

    ACC = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    SEN = (TP) / (TP + FN + 1e-6)
    SPE = (TN) / (TN + FP + 1e-6)
    PPV = (TP) / (TP + FP + 1e-6)
    F_score = (2 * SEN * PPV) / (SEN + PPV + 1e-6)
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)+ 1e-6)
    
    if probs != None:
        return [AUC, ACC, SEN, SPE, F_score, MCC]
    else:
        return [ACC, SEN, SPE, F_score, MCC]



if __name__ == "__main__":
    gt, pred = np.random.rand(10, 1, 100, 100), np.random.rand(10, 1, 100, 100)
    print('MSE', mse(gt, pred))
    print('NMSE', nmse(gt, pred))
    print('PSNR', psnr(gt, pred))
    print('SSIM', ssim(gt, pred))
    print('MI', mi(gt, pred))
