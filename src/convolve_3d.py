import numpy as np
import os
from Line_Filter_Transform import Euler_Angles_Sphere
import argparse 
from Fast_Marching_Cython import Convolution_3D
from Func_Norm_Utils import P_Norm_Normalization
from numpy import inf 

parser = argparse.ArgumentParser(description='3D Convolution with a kernel on SE(3)')

parser.add_argument("OCTA_numpy",
    type=str,
    help="string_path_to_Orientation_Score_Volume_in_npy_format")
parser.add_argument("Angle_Number",
    type=int,
    help="Number_of_Angles_for_Convolution")
parser.add_argument("D_33",
    type=np.double,
    help="Diffusion_Coef_for_spatial_regularization")
parser.add_argument("D_44",
    type=np.double,
    help="Diffusion_Coef_for_angular_regularization")
parser.add_argument("Int_Time",
    type=np.double,
    help="Integration_time_of_Diffusion")
parser.add_argument("Kernel_Size",
    type=int,
    help="Size_of_rectangular_Window_on_SE(3)")
parser.add_argument("Angles_conv",
    type=int,
    help="Number_of_Nearaest_Orientation_for_Convolution")
parser.add_argument("--Method",
    type=str,
    help="Kernel_Approximation_Methods",
    default = 'Kernel_2D')
args = parser.parse_args()

assert type(args.Method) == str, 'Available Methods: [Kernel_2D,Mises_Fischer_Kernel,Contour_Enh,Contour_Compl]'

Or_Score = np.load(args.OCTA_numpy).astype(np.double)
Fibonacchi_Euler_Angles,Fibonacchi_Euler_Points = Euler_Angles_Sphere(samples=args.Angle_Number)

' Convolve_Volume '

Or_Score = P_Norm_Normalization(Or_Score,p = 2)
print(Or_Score.shape)
print(Fibonacchi_Euler_Angles.shape[0])

' Explicit Kernels '

if args.Method == 'Kernel_2D' or args.Method == 'Mises_Fischer_Kernel':
    Conv_Vol = Convolution_3D.convolution_routine(Or_Score,args.D_33,args.D_44,args.Int_Time,args.Kernel_Size,Fibonacchi_Euler_Points,Fibonacchi_Euler_Angles,args.Angles_conv,Method = args.Method) 
elif args.Method == 'Contour_Enh' or args.Method == 'Contour_Compl':
    Conv_Vol = Convolution_3D.convolution_routine_stochastic(Or_Score,args.D_33,args.D_44,args.Int_Time,args.Kernel_Size,Fibonacchi_Euler_Points,Fibonacchi_Euler_Angles,args.Angles_conv,Method = args.Method)


Conv_Vol = P_Norm_Normalization(np.array(Conv_Vol),p = 2)

' Save Volume '

np.save(os.path.join('Data_Folder/','Conv_Vol_'+str(args.Angles_conv)+'_'+str(args.Kernel_Size)+'_'+args.Method),np.array(Conv_Vol).astype(np.float32))