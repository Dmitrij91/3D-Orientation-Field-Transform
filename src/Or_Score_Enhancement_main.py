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
from Fast_Marching_Cython import Line_Filter_Transform_Cython
import Line_Filter_Transform
from Fast_Marching_Cython import Orientation_Score_Enhancment_Filter

parser = argparse.ArgumentParser(description='OCTA_Volume_Preprocessing_Routine_for_Vessel_Enhancement_on_SE(3)')

parser.add_argument("Or_Score",
    type=str,
    help="Path_to_File_after_Preprocessing_with_Orientation_Score_Diffusion")
parser.add_argument("--Angle_Max",
    type=int,
    help="Size_of_a_Patch_at_each_Voxel_where_to_seek_for_the_direction_with_maximal_Response",
    default=70)
parser.add_argument("--Angle_Num",
    type=int,
    help="Number_of_used_direction_for_Orientation_Score_Transform",
    default=10)
parser.add_argument("--Radius",
    type=int,
    help="Maximal Radius in voxels",
    default=8)
parser.add_argument("--Angle_Res",
    type=int,
    help="Resolution of angles for tubulatory measure",
    default=10)
args = parser.parse_args()


Test = np.load(args.Or_Score)
_,Dir_Array = Line_Filter_Transform.Euler_Angles_Sphere(args.Angle_Num)
Enh = Orientation_Score_Enhancment_Filter.Enhancement_Filter_Orientation_Score(Test[0:100,0:100,0:100,:].astype(np.float64),Dir_Array,args.Radius,args.Angle_Res,0.2)
Enh_angle = Orientation_Score_Enhancment_Filter.Filter_Enh_angle_sum(Enh,args.Angle_Max)
np.save(os.path.join("Data_Folder/","Orientation_Score_Enhanced"),np.array(Enh_angle).astype(np.float32))