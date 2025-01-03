import numpy as np
import OOF_Func 
import argparse
from scipy import ndimage
import os

parser = argparse.ArgumentParser(description=' Maximal Radius Filter ')

parser.add_argument("OCTA_numpy",
    type=str,
    help="string_path_to_Volume_in_npy_format")

parser.add_argument("List_Rad",
    type=list,
    help="List_Vessel_Radius")

parser.add_argument("h",
    type=str,
    help="Discretization_of_Radius_Interval")
args = parser.parse_args()


Scale_size = args.h

Filter_Volume_OCT = []

for radius in Radius_Vessels:
        
    rsp,_,_  = OOF_Func.response(ndimage.filters.gaussian_filter(Test_Oct_Cropped,sigma = 1),radii = np.linspace(radius,radius+args.h,10),rsptype='oof')
    Filter_Volume_OCT.append(rsp)

Max_Radius_Vol = np.max(np.array(Filter_Volume_OCT),axis = 0)
np.save(os.path.join('Data_Folder/','Radius_Max'),Max_Radius_Vol.astyoe(np.float32))