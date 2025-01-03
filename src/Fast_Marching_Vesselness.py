import numpy as np
import os
import argparse 
from Distance_Utilities import dist
from Graph_Build import adj_matrix
from Fast_Marching_Cython import Fast_Marching_Binary_Heap
from Fast_Marching_Cython import Fast_Marching_Non_Local_Tools
from Fast_Marching import Eikonal_Eq_Solve
from Graph_Build import adj_matrix_adaptive_Mod
from scipy import ndimage
from scipy.sparse import csr_matrix
from Graph_Build import Entropy
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='Postproccesing enhancement step for diffused data on (SE(3)) ')

parser.add_argument("Volume",
    type=str,
    help="Path_to_Volume_in_npy_format")
parser.add_argument("Energy",
    type=str,
    help="Path_to_Energy_in_npy_format")
parser.add_argument("Input_Sigma",
    type=np.double,
    help="Non_Local_Means_Scaling")
parser.add_argument("Input_Scale",
    type=np.double,
    help="Scale_Data_within_Adj_Matrix")
args = parser.parse_args()


Volume = np.load(args.Volume).astype(np.double)[0:146,0:146,0:146,:]
Energy = np.load(args.Energy).astype(np.double)
Volume = np.max(Volume,axis = 3)

' Get Indices for Vessel Seeds based on Raw OCTA Data'

def Get_Pixel(Input_Image,tau):
    Data = Input_Image.copy()/(Input_Image.max())
    Data[Data > 1] = 0
    Data[Data < tau] = 0
    Vessel_Mask = Data.astype(bool)
    Indices = np.where(Vessel_Mask.reshape(-1) != 0)[0].astype(np.int32)
    print('Threshold_Level'+str(Indices.shape[0]/Input_Image.reshape(-1).shape[0]*100))
    return Indices

def Nonlocal_Sigma_Par(f_0,shape):
    
    ' Can be modified to patches '
    
    win_mean = ndimage.uniform_filter(f_0, shape)
    win_sqr_mean = ndimage.uniform_filter(f_0**2, shape)
    win_var = win_sqr_mean - win_mean**2
    
    return win_var

shape_vol = Volume.shape


Ad = adj_matrix_adaptive_Mod(ndimage.filters.gaussian_filter(Volume,sigma = args.Input_Scale),(3,3,3),sigma=args.Input_Sigma)

Image = np.array([Entropy(Ad.data[Ad.indptr[k]:Ad.indptr[k+1]]) for k in range(len(Ad.indptr)-1)])
H = Image.reshape(Volume.shape)

#sigma_weights = (Nonlocal_Sigma_Par(Volume,Volume.shape)**(1/2)).reshape(-1)

#sigma_weights = np.ones_like(sigma_weights.shape)

#data,indices,indptr = Fast_Marching_Non_Local_Tools.Non_Local_Means_cython_3D((Volume),1/sigma_weights,Ad.indptr,Ad.indices,shape_vol,sigma = 20,window_size = 3)


#np.save('Data_Folder/Sparse_Mat',mat)



' Show Entropy '
    
Image_non_local = np.array([Entropy(Ad.data[Ad.indptr[k]:Ad.indptr[k+1]]) for k in range(len(Ad.indptr)-1)])
H_non_local = Image_non_local.reshape(Volume.shape)

plt.figure(dpi = 200)
plt.subplot(1,3,1)
plt.imshow(H[60,:,:],cmap = 'gnuplot')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(H_non_local[60,:,:],cmap = 'gnuplot')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(Volume[60,:,:])
plt.axis('off')
plt.tight_layout()
plt.show()
plt.savefig('Data_Folder/Visualization_Fast_Marching/Show_Entropy_Fast_Marching')


Indices = Get_Pixel(Volume,0.8)
print(Indices)

Test_Active_List,Test_Active_Val = Fast_Marching_Binary_Heap.Eikonal_Eq_Solve_Cython(Energy.reshape(-1),Ad.data,Ad.indices,\
                                                           Ad.indptr,Indices[0:1],np.array([3,3,3]).astype(np.intc))
#Test_Active_List,Test_Active_Val = Eikonal_Eq_Solve((np.exp(-Volume)/(np.max(np.exp(-Volume),axis = 0)[None,:,:])).reshape(-1),Ad,Indices)
np.save('Data_Folder/Tracked_Volume',Test_Active_List)