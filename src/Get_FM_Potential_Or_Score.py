import numpy as np
import os
from Line_Filter_Transform import Euler_Angles_Sphere_2
import argparse 
from Func_Norm_Utils import P_Norm_Normalization
from Fast_Marching_Cython import Orientation_Score_Enhancment_Filter,Fast_Marching_Energy
from Fast_Marching_CUDA import Fast_Marching_Energy_Cuda

parser = argparse.ArgumentParser(description=' Compute radius and orientation dependent Potential for Fast_Marching Routine ')

parser.add_argument("--Data_Or",
    type=str,
    help="string_path_to_Orientation_Score_Volume_in_npy_format")
parser.add_argument("--Angle_Number",
    type=int,
    help="Number_of_Orienations"),
parser.add_argument("--Theta_num",
    type=int,
    help="Number_Of_Angles_For_Sampling_on_Circle_orthogonal_to_particular_Orientation")
parser.add_argument("Method",
    type=str,
    help="Possible_Computation_devices [CPU,GPU_Cuda]",
    default = 'CPU')
args = parser.parse_args()

Rad_par = (0.3,3,10)
Rad_Arr = np.linspace(*Rad_par)

_,Fibonacchi_Euler_Points = Euler_Angles_Sphere_2(samples=args.Angle_Number)

Fibonacchi_Euler_Points = Fibonacchi_Euler_Points

Or_Score = np.load(args.Data_Or)[0:146,0:146,0:146,:]
if args.Method == 'GPU':
    Test = Fast_Marching_Energy_Cuda.energy_radius(Or_Score.astype(np.float32),Rad_Arr.astype(np.float32),\
                    Fibonacchi_Euler_Points.astype(np.float32),Rad_par[0],Rad_par[2],args.Theta_num)
elif args.Method == 'CPU':
    Test = Fast_Marching_Energy.Energy_Radius(Or_Score.astype(np.double),Rad_Arr,Fibonacchi_Euler_Points,Rad_par[0],Rad_par[2],args.Theta_num)

np.save('FM_Test',np.array(Test).astype(np.float32))