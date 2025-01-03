import numpy as np
import argparse
import os.path
import sys
import vol_viz_OCTA
from vol_viz_OCTA import OCTScan
from vol_viz_OCTA import draw
import h5py


parser = argparse.ArgumentParser(description='Convert_Vol_to_HDF5')

parser.add_argument("OCTA_File",
    type=str,
    help="Path_to_File")
parser.add_argument("--format",
    type=str,
    help="File_Format")

args = parser.parse_args()

assert os.path.isfile(args.OCTA_File), f"File {args.OCTA_File} not found."
assert args.OCTA_File.endswith(".vol")   

file = open(args.OCTA_File, "rb").read()
oct = OCTScan(file)
oct.filename = args.OCTA_File.split(os.path.sep)[-1]

'Load OCTA Data'

X = oct.headerinfo.SizeX
Y = oct.headerinfo.NumBScans
Z = oct.headerinfo.SizeZ


octdata_full  = np.zeros((Z,X,Y)) 

for k in range(Y):
    octdata_full[:,:,k] = ((oct.bscans)[k]).data
    
    ' Remove_Scanner_Artifacts '

octdata_full[octdata_full > 10] = 1e-10
octdata_full[octdata_full <= 0] = 1e-10


#if args.format == 'hdf5':

h5f = h5py.File(os.path.join("Data_Folder/","OCTA_Data_Set"), 'w')
    
h5f.create_dataset("OCTA_Data_Set",data = octdata_full,dtype= np.float32)

h5f.close()

