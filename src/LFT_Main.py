import importlib
import numpy as np
import argparse
import os.path
import sys
from Line_Filter_Transform import*
from Distance_Utilities import dist
from Graph_Build import adj_matrix
from Fast_Marching_Cython import Line_Filter_Transform_Cython

parser = argparse.ArgumentParser(description='OCTA_Volume_Preprocessing_Routine_for_Vessel_Enhancement')

parser.add_argument("OCTA_File",
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
assert os.path.isfile(args.OCTA_File), f"File {args.OCTA_File} not found."
assert args.OCTA_File.endswith(".npy")

GAMMA = 4


octdata_full_Test = np.load(args.OCTA_File)


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