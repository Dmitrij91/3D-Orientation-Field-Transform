import numpy as np 
from skimage import exposure, util
import imageio as io
import matplotlib.pyplot as plt 


' Different Normalization Routines on Orientation Score Domain in 3D '

def Histogram_Eq(Image,perc):
    Ret_Norm = np.zeros(Image.shape)

    Ret_Norm[:,:,:] = np.clip(im[:,:,:,k],
                        np.percentile(Image[:,:,:], perc),
                        np.percentile(Image[:,:,:], 100-perc))
    Ret_Norm[:,:,:] = (Ret_Norm[:,:,:,k] - Ret_Norm[:,:,:].min()) / (Ret_Norm[:,:,:].max() - Ret_Norm[:,:,:].min())

    sigmoid = np.exp(-3 * np.linspace(0, 1, Ret_Norm[:,:,:].shape[0]))
    im_degraded = (Ret_Norm[:,:,:].T * sigmoid).T

    Ret_Norm[:,:,:,k], im_degraded_he = [exposure.equalize_hist(im) for im in [Ret_Norm[:,:,:,k], im_degraded]]

    return Ret_Norm


def Monotone_Gray_Value_Transform_max(Image,q):

    assert len(Image.shape) == 4

    return (np.max((Image - Image.min())**(q),axis = 3)-\
        np.min(np.max((Image - Image.min())**(q),axis = 3),axis = (1,2))[:,None,None])/\
        (np.max(np.max((Image - Image.min())**(q),axis = 3),axis = (1,2))[:,None,None]-\
        np.min(np.max((Image - Image.min())**(q),axis = 3),axis = (1,2))[:,None,None])

def Monotone_Gray_Value_Transform_sum(Image,q):

    assert len(Image.shape) == 4

    return (np.sum((Image - Image.min())**(q),axis = 3)-\
        np.min(np.sum((Image - Image.min())**(q),axis = 3),axis = (1,2))[:,None,None])/\
        (np.max(np.sum((Image - Image.min())**(q),axis = 3),axis = (1,2))[:,None,None]-\
        np.min(np.sum((Image - Image.min())**(q),axis = 3),axis = (1,2))[:,None,None])

def P_Norm_Normalization(Or_Score,p):

    Score_Norm = Or_Score - Or_Score.min()[None,None,None,None]

    Score_Norm = Score_Norm/((np.einsum('ijkl->l',Score_Norm**p))[None,None,None,:]**(1/p))

    return Score_Norm

def Exp_Lifting(Image,sigma):

    return np.exp(Image/sigma)/(np.einsum('ijkl -> ijk',np.exp(Image/sigma))[:,:,:,None])

def Norm_Matplotlib(Or_Score_Vol):
    
    Vol_Norm = np.max(Or_Score_Vol,axis = 3)

    for k in range(Vol_Norm.shape[0]):
        Z_Slice = Vol_Norm[k,:,:]
        norm = plt.Normalize(vmin = Z_Slice.min(),vmax = Z_Slice.max())
        cmap = plt.cm.gray
        Vol_Norm[k,:,:] = rgb2gray(cmap(norm(Z_Slice)))

    return Vol_Norm