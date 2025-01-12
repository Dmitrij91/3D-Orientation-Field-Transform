import numpy as np
import argparse
import os.path
import sys
from scipy.fftpack import fftn,fftshift,ifftn,ifftshift
from scipy.ndimage import zoom
import multiprocessing
import tempfile
from joblib import Parallel, delayed,load,dump
import shutil

parser = argparse.ArgumentParser(description='Orientation_Score_Transform_3D')

parser.add_argument("OCTA_File",
    type=str,
    help="Path_to_File")
parser.add_argument("--window",
    type=int,
    help="Rectangular_Size_For_Padding",
    default=400)
parser.add_argument("--Wavelet_Size",
    type=int,
    help="Wavelet_Size_of_Filter_Mask",
    default=100)

args = parser.parse_args()
assert os.path.isfile(args.OCTA_File), f"File {args.OCTA_File} not found."
assert args.window%args.Wavelet_Size == 0  
assert args.OCTA_File.endswith(".npy")   

' Functions_for_Multiple_Processing_For_Loops '
Cones_List = []

def Interpolate_Cones(k,Cones,Var_Zoomed,Pad_Array,Cones_List):
    print(f'finished "----{k}----" Mask')
    Cones_List.append(k)
    Cones_Real = zoom((Cones)[k].real.reshape(args.Wavelet_Size,args.Wavelet_Size,args.Wavelet_Size),(Var_Zoomed,Var_Zoomed,Var_Zoomed),order = 3)
    Cones_Imag = zoom((Cones)[k].real.reshape(args.Wavelet_Size,args.Wavelet_Size,args.Wavelet_Size),(Var_Zoomed,Var_Zoomed,Var_Zoomed),order = 3)
    Cones_Zoomed = Cones_Real+1j*Cones_Imag

    return fftshift(fftn(Pad_Array))*(Cones_Zoomed)

def Inverse_Fourier_Assign(k,Angle_Vector,Cones_List):

    print(f'finished "----{k}----" Mask')
    Cones_List.append(k)
    return np.real(ifftn(ifftshift(Angle_Vector[:,:,:,k])))

file = open(args.OCTA_File, "rb").read()

oct_array = np.load('Data_Folder/Test_OCTA_Data.npy')

' Load Predefined Filter Masks '

Cones = np.load('Filter_Mask_Orientation_Score_3D/Wavelet_Filter_new.npy')


' Perform Transformation '

X,Y,Z = oct_array.shape

Pad_Array = oct_array.copy()

' Define_Padding_Dimensions '

if X <= args.window:

    if (args.window-X)%2 == 0:
        
        pad_X = (int((args.window-X)/2),int((args.window-X)/2))

    elif (args.window-X)%2 == 1:

        pad_X = (np.rint((args.window-X)/2),np.rint((args.window-X)/2)+1)

    Pad_Array = np.pad(Pad_Array,[pad_X,(0,0),(0,0)],'constant')

else:

    if (X-args.window)%2 == 0:

        Cut_Borders_X = (int((X-args.window)/2),int((X-args.window)/2))

    elif (X-args.window)%2 == 1:

        Cut_Borders_X = (np.rint((X-args.window)/2),np.rint((X-args.window)/2)+1)

    Pad_Array = Pad_Array[Cut_Borders_X[0]:X-Cut_Borders_X[1],:,:]

if Y <= args.window:

    if (args.window-Y)%2 == 0:
        
        pad_Y = (int((args.window-Y)/2),int((args.window-Y)/2))

    elif (args.window-Y)%2 == 1:

        pad_Y = (np.rint((args.window-Y)/2),np.rint((args.window-Y)/2)+1)

    Pad_Array = np.pad(Pad_Array,[(0,0),pad_Y,(0,0)],'constant')

else:

    if (Y-args.window)%2 == 0:

        Cut_Borders_Y = (int((Y-args.window)/2),int((Y-args.window)/2))

    elif (Y-args.window)%2 == 1:

        Cut_Borders_Y = (np.rint((Y-args.window)/2),np.rint((Y-args.window)/2)+1)

    Pad_Array = Pad_Array[:,Cut_Borders_Y[0]:Y-Cut_Borders_Y[1],:]

if Z <= args.window:

    if (args.window-Z)%2 == 0:
        
        pad_Z = (int((args.window-Z)/2),int((args.window-Z)/2))

    elif (args.window-Z)%2 == 1:

        pad_Z = (np.rint((args.window-Z)/2),np.rint((args.window-Z)/2)+1)

    Pad_Array = np.pad(Pad_Array,[(0,0),(0,0),pad_Z],'constant')

else:

    if (Z-args.window)%2 == 0:

        Cut_Borders_Z = (int((Z-args.window)/2),int((Z-args.window)/2))

    elif (X-args.window)%2 == 1:

        Cut_Borders_Z = (np.rint((Z-args.window)/2),np.rint((Z-args.window)/2)+1)

    Pad_Array = Pad_Array[:,:,Cut_Borders_Z[0]:Z-Cut_Borders_Z[1]]

Var_Zoomed = int(args.window/args.Wavelet_Size)

num_cores = multiprocessing.cpu_count()
Cones_Iterator = range(len(Cones))

Show = Parallel(n_jobs=num_cores,backend="threading")(delayed(Interpolate_Cones)(iterate,Cones,Var_Zoomed,Pad_Array,Cones_List) for iterate in Cones_Iterator)

print(Cones_List)
print(np.array(Show).shape)

Image = np.zeros((args.window,args.window,args.window,len(Cones)))

' Store Angle_Vector in Cache Directory to speed up subsequent paralell computations '

folder = 'Data_Folder/Cache/Cache_Directory_OCTA_Diffusion_win_size_'+str(args.window)+'wave_size'+str(args.Wavelet_Size)

Cones_List = np.argsort(np.array(Cones_List))

if os.path.exists(folder+'/Stored_Angle_Vector'):
    print('Loading_Already_Existed_Cache')
    data_angle_vector = os.path.join(folder, 'Stored_Angle_Vector')
    Angle_Vector = load(data_angle_vector, mmap_mode='r')
else: 
    print(' Set_Up_Cache_for_parallel_Computation')
    os.makedirs(folder,exist_ok=True)

    Angle_Vector = np.zeros((args.window,args.window,args.window,len(Cones)),dtype = np.complex128)

    Angle_Vector[:,:,:,:] = np.moveaxis(np.array(Show)[Cones_List][:,:,:,:],0,-1)
    print('Save_Cache')
    data_angle_vector = os.path.join(folder, 'Stored_Angle_Vector')
    dump(Angle_Vector, data_angle_vector)
    Angle_Vector = load(data_angle_vector, mmap_mode='r')

Cones_List = []

Image = Parallel(n_jobs=num_cores,backend="threading")(delayed(Inverse_Fourier_Assign)(item,Angle_Vector,Cones_List) for item in Cones_Iterator)    

del(Angle_Vector)

try:
    shutil.rmtree(folder)
except:  
    print('Could not clean-up automatically.')

print(Cones_List)

Image = np.array(Image)[np.argsort(np.array(Cones_List))]

Image = np.moveaxis(np.array(Image),0,-1)

' Response Normalization '

#Image = Image-np.min(Image[:,:,:,:],axis = 3)[:,:,:,None]
#Image = (Image[:,:,:,:]/(np.max(Image[:,:,:,:],axis = 3)[:,:,:,None]))**2

np.save(os.path.join("Data_Folder/","Orientation_Score_Data_win_size_"+str(args.window)+'wave_size'+str(args.Wavelet_Size)),Image.astype(np.float32))


'Save_Enhanced_Volume after Max Mononote Grayvalue Transfomration Response Normalization '

Oct_max_response = (np.max((Image - Image.min())**(0.25),axis = 3)-\
        np.min(np.max((Image - Image.min())**(0.25),axis = 3),axis = (1,2))[:,None,None])/\
(np.max(np.max((Image - Image.min())**(0.25),axis = 3),axis = (1,2))[:,None,None]-\
        np.min(np.max((Image - Image.min())**(0.25),axis = 3),axis = (1,2))[:,None,None])


np.save(os.path.join("Data_Folder/","octdata_full_max_Orientation_Score_win_size_"+str(args.window)+'wave_size'+str(args.Wavelet_Size)),Oct_max_response.astype(np.float32))