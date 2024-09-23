import importlib
import numpy as np
import argparse
import os.path
import sys
from Line_Filter_Transform import*
import vol_viz_OCTA
from vol_viz_OCTA import OCTScan
from vol_viz_OCTA import draw
from Distance_Utilities import dist
from Graph_Build import adj_matrix
importlib.reload(vol_viz_OCTA)
from Fast_Marching_Cython import Line_Filter_Transform_Cython

parser = argparse.ArgumentParser(description='OCTA_Volume_Preprocessing_Routine_for_Vessel_Enhancement')

parser.add_argument("OCTA_Vol_File",
    type=str,
    help="Path_to_File")
parser.add_argument("--Patch_size",
    type=tuple,
    help="Size_of_a_Patch_at_each_Voxel_where_to_seek_for_the_direction_with_maximal_Response",
    default=(3,3,3))
parser.add_argument("--SizeX",
    type=int,
    help="Size_of_Volume_Crop_in_B_Scan_Direction",
    default=400)
parser.add_argument("--SizeZ",
    type=int,
    help="Size_of_Volume_Crop_in_A_Scan_Direction",
    default=400)
parser.add_argument("--NumBScans",
    type=int,
    help="Number_Bscans",
    default=400)

args = parser.parse_args()
assert os.path.isfile(args.OCTA_Vol_File), f"File {args.OCTA_Vol_File} not found."
assert args.OCTA_Vol_File.endswith(".vol")

GAMMA = 4


file = open(args.OCTA_Vol_File, "rb").read()
oct = OCTScan(file)
oct.filename = args.OCTA_Vol_File.split(os.path.sep)[-1]

'Load OCTA Data'

X = oct.headerinfo.SizeX
Y = oct.headerinfo.NumBScans
Z = oct.headerinfo.SizeZ

octdata_full  = np.zeros((Z,X,Y)) 
segments_full = np.zeros((oct.bscans[0].segments.shape[0], X, Y))
for k in range(Y):
    octdata_full[:,:,k] = ((oct.bscans)[k]).data
    segments_full[:,:,k] = ((oct.bscans)[k]).segments
    
    ' Remove_Scanner_Artifacts '

octdata_full[octdata_full > 10] = 1e-10
octdata_full[octdata_full <= 0] = 1e-10

# Slice Volume

octdata_full_Test = octdata_full[47:47+args.SizeZ,56:56+args.SizeX,56:56+args.NumBScans] 

'Save_Cropped_Volume'

#np.save(os.path.join("Data_Folder/","octdata_full_Test"), octdata_full_Test.astype(np.float32))

' Uniform Discretization of the unit Sphere for by Euler angles '

_,Fibonacchi_Points = Euler_Angles_Sphere(samples=60)

' Get Volume directions on integer grid '

OCTA_Volume_Coord = Get_Coordinates(octdata_full_Test)

' Extract_Adjacency Matrix '

Ad = adj_matrix(octdata_full_Test, shape_img=args.Patch_size)

' Perform_Line_Filter_Transform '

Test_enhanced_OCTA = Line_Filter_Transform_Cython.Main_Line_Filter_Transform_Cython(octdata_full_Test.reshape(-1),OCTA_Volume_Coord.reshape(-1,3).reshape(-1).astype(np.int32)\
,Ad.indices,Ad.indptr,Fibonacchi_Points.reshape(-1),np.array([args.Patch_size[0],args.Patch_size[1],args.Patch_size[2]],dtype = np.int32))

'Save_Enhanced_Volume'

np.save(os.path.join("Data_Folder/","octdata_full_Test_LFT_Enhanced"), Test_enhanced_OCTA.reshape(args.SizeZ,args.SizeX,args.NumBScans,4).astype(np.float32))