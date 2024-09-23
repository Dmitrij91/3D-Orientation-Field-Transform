import importlib
import numpy as np
import argparse
import os.path
import sys
from Line_Filter_Transform import*
from Distance_Utilities import dist
from Graph_Build import adj_matrix
from Cython_OFT import Line_Filter_Transform_Cython

parser = argparse.ArgumentParser(description='Preprocessing_Routine_for_Line_Enhancement')

parser.add_argument("Vol_File",
    type=str,
    help="Path_to_File")
parser.add_argument("--Patch_size",
    type=tuple,
    help="Size_of_a_Patch_at_each_Voxel_where_to_seek_for_the_direction_with_maximal_Response",
    default=(3,3,3))
parser.add_argument("--SizeX",
    type=int,
    help="Size_of_Volume_X_Direction",
    default=400)
parser.add_argument("--SizeZ",
    type=int,
    help="Size_of_Volume_in_Z_Direction",
    default=400)
parser.add_argument("--SizeY",
    type=int,
    help="Size_of_Volume_in_Y_Direction",
    default=400)

args = parser.parse_args()
assert os.path.isfile(args.OCTA_Vol_File), f"File {args.Vol_File} not found."
assert args.Vol_File.endswith(".npy")

Vol  = np.zeros((X,Y,Z)) 
    
' Remove_Scanner_Artifacts '

octdata_full[octdata_full > 10] = 1e-10
octdata_full[octdata_full <= 0] = 1e-10

' Uniform Discretization of the unit Sphere for by Euler angles '

_,Fibonacchi_Points = Euler_Angles_Sphere(samples=60)

' Get Volume directions on integer grid '

Volume_Coord = Get_Coordinates(Vol)

' Extract_Adjacency Matrix '

Ad = adj_matrix(Vol, shape_img=args.Patch_size)

' Perform_Line_Filter_Transform '

Enhanced_Vol = Line_Filter_Transform_Cython.Main_Line_Filter_Transform_Cython(Vol.reshape(-1),Volume_Coord.reshape(-1,3).reshape(-1).astype(np.int32)\
,Ad.indices,Ad.indptr,Fibonacchi_Points.reshape(-1),np.array([args.Patch_size[0],args.Patch_size[1],args.Patch_size[2]],dtype = np.int32))

'Save_Enhanced_Volume'

np.save(os.path.join("../data","Test_LFT_Enhanced"), Enhanced_Vol.reshape(args.SizeZ,args.SizeX,args.NumBScans,4).astype(np.float32))
