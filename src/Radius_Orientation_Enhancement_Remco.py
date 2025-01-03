import numpy as np
import os
from Line_Filter_Transform import Euler_Angles_Sphere,Euler_Angles_Sphere_2
import argparse 
from Func_Norm_Utils import P_Norm_Normalization
from Fast_Marching_Cython import Orientation_Score_Enhancment_Filter

parser = argparse.ArgumentParser(description='Postproccesing enhancement step for diffused data on (SE(3)) ')

parser.add_argument("Or_Score",
    type=str,
    help="string_path_to_Orientation_Score_Volume_in_npy_format")
parser.add_argument("Angle_Number",
    type=int,
    help="Number_of_Orienations")
parser.add_argument("Rad_num",
    type=int,
    help="Radius_Discretization")
parser.add_argument("Theta_num",
    type=int,
    help="Number_Of_Angles_For_Sampling_on_Circle_orthogonal_to_particular_Orientation")
parser.add_argument("Rad_Space",
    type= np.double,
    help="Spatial_Circle_Spacing")
parser.add_argument("Angle_Prec",
    type=int,
    help="Subdivision_of_[0,2pi]_for_miniminzation")
args = parser.parse_args()


_,Fibonacchi_Euler_Points = Euler_Angles_Sphere(samples=args.Angle_Number)

Fibonacchi_Euler_Points = Fibonacchi_Euler_Points.astype(np.double)

print(Fibonacchi_Euler_Points.shape)

Or_Score = np.load(args.Or_Score).astype(np.double)[0:150,0:150,0:150,:]
print(Or_Score.shape)

' Process_Volume '

print('___Start_First_Iteration__')

Enh_Data = Orientation_Score_Enhancment_Filter.Enhancement_Filter_Orientation_Score(Or_Score,Fibonacchi_Euler_Points,\
                                                                args.Rad_num,args.Theta_num,args.Rad_Space)

print('___Start_Second_Iteration__')

Enh_Data = Orientation_Score_Enhancment_Filter.Filter_Enh_angle_sum(Enh_Data,args.Angle_Prec,args.Theta_num)


print('___Enhancement_Finished__')

' Save Volume '

np.save(os.path.join('Data_Folder/','OCTA_Radius_Enhanced'),np.array(Enh_Data).astype(np.float32))