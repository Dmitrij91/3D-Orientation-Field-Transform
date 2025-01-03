import numpy as np
import argparse
import os.path
import sys
import vol_viz_OCTA
from vol_viz_OCTA import OCTScan
from vol_viz_OCTA import draw
from Distance_Utilities import dist
from Graph_Build import adj_matrix
from Fast_Marching_Cython import Fast_Marching_Binary_Heap
from Fast_Marching_Cython import Fast_Marching_Non_Local_Tools
from Graph_Build import adj_matrix_adaptive_Mod
from scipy.sparse import csr_matrix
from Graph_Build import Entropy
from Func_Norm_Utils import P_Norm_Normalization



parser = argparse.ArgumentParser(description='AF_Flow_Segmentation_via_Fast_Marching_Avaraging')

parser.add_argument("--OCTA_File",
    type=str,
    help="Path_to_File")
parser.add_argument("--OCTA_Mask",
    type=str,
    help="Path_to_File"),
parser.add_argument("--Diffused_OCTA",
    type=str,
    help="Path_to_Diffused_Data_in_npy_format"),
parser.add_argument("--window",
    type=int,
    help="Avaraging_window",
    default=400)
parser.add_argument("--av_window",
    type=tuple,
    help="Avaraging_window",
    default=(3,3,3))

args = parser.parse_args()
assert os.path.isfile(args.OCTA_File), f"File {args.OCTA_File} not found."
assert args.window%100 == 0  
assert args.Diffused_OCTA.endswith(".npy")  

def Get_Pixel(Input_Image,tau):
    Data = Input_Image.copy()/(Input_Image.max())
    Data[Data > 1] = 0
    Data[Data < tau] = 0
    Vessel_Mask = Data.astype(bool)
    Indices = np.where(Vessel_Mask.reshape(-1) != 0)[0].astype(np.int32)
    print('Threshold_Level'+str(Indices.shape[0]/Input_Image.reshape(-1).shape[0]*100))
    return Indices

Volume_Diffused = np.load(args.Diffused_OCTA)

OCTA_Vol = np.load(args.OCTA_File)
OCTA_Vol_mask = np.load(args.OCTA_Mask)

Mask = np.zeros_like(OCTA_Vol_mask)

Mask[OCTA_Vol_mask == 0] = 1
Mask[OCTA_Vol_mask == 8] = 1
Mask[OCTA_Vol_mask == 13] = 1

Volume_Diffused[Mask.astype(bool)] = 0
Vol_Diffused_norm = P_Norm_Normalization(Volume_Diffused,2) 
Indices = Get_Pixel(OCTA_Vol,1)
Mask = np.max(Vol_Diffused_norm,axis = 3)


Ad = adj_matrix_adaptive_Mod(Mask,(3,3,3),sigma=0.2)

#Image = np.array([Entropy(Ad.data[Ad.indptr[k]:Ad.indptr[k+1]]) for k in range(len(Ad.indptr)-1)])
#H = Image.reshape(Volume.shape)


Test_Active_List,Test_Active_Val = Fast_Marching_Binary_Heap.Eikonal_Eq_Solve_Cython(np.exp(-Mask.astype(np.double)/0.5).reshape(-1),Ad.data.astype(np.double),Ad.indices.astype(np.intc),\
                                                           Ad.indptr.astype(np.intc),Indices.astype(np.intc),np.array([3,3,3]).astype(np.intc))
#Test_Active_List,Test_Active_Val = Eikonal_Eq_Solve((np.exp(-Volume)/(np.max(np.exp(-Volume),axis = 0)[None,:,:])).reshape(-1),Ad,Indices)

s = 0
F = np.zeros(OCTA_Vol.reshape(-1).shape)
for i  in Test_Active_List:
    F[i] = s
    s += 1

np.save('/scratch/dmitrij/Datafolder/Fast_Marching_OCTA'+args.Diffused_OCTA,F.reshape(OCTA_Vol.shape)) 
np.save('/scratch/dmitrij/Data_Folder/Tracked_OCTA_Vol',Test_Active_List)


