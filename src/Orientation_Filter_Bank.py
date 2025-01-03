import numpy as np
import os
import argparse
from Spherical_Func import Radial_Erf, polar_from_cartesian
from Line_Filter_Transform import Euler_Angles_Sphere
from Fast_Marching_Cython import Wigner_D_Function_Cython
from Line_Filter_Transform import Euler_Angles_Sphere
import time

parser = argparse.ArgumentParser(description="Create_Filter_Masks_For_Orientation_Score_Transform_in_Fourier_Domain")
parser.add_argument("--Num_Angles",
    type=int,
    help="Number_of_uniform_Euler_angles",
    default=30)
parser.add_argument("--Grid_Size",
    type=int,
    help="Rectangular_Spatial_Dimension_X,Y,Z",
    default=100)
args = parser.parse_args()

assert args.Grid_Size%2 == 0, "The grid size Grid_Size must be an even number"

x = np.linspace(-int(args.Grid_Size/2),int(args.Grid_Size/2),args.Grid_Size)
y = np.linspace(-int(args.Grid_Size/2),int(args.Grid_Size/2),args.Grid_Size)
z = np.linspace(-int(args.Grid_Size/2),int(args.Grid_Size/2),args.Grid_Size)

X,Y,Z = np.meshgrid(x,y,z)

Coordinates_as_vector       = np.array([X,Y,Z]).reshape(3,-1)

' Array_of_uniformly_spaces_Euler_angles_symmetric_to_z_axis '

Fibonacchi_Euler_Angles,_ = Euler_Angles_Sphere(samples=args.Num_Angles)

vec_cor_1     = np.swapaxes(np.array([X,Y,Z]).reshape(3,Coordinates_as_vector.shape[1]),axis1=1,axis2=0)

pol_cor = polar_from_cartesian(vec_cor_1)

Cones = []

Radial_Part = Radial_Erf(pol_cor,gamma= 0.87,sigma_erf= 2)

k = 1
print('Start_Iterations')
for array in Fibonacchi_Euler_Angles:

    Time_Start = time.time()

    Get_Kernel_Rot     = Wigner_D_Function_Cython.Spherical_Kernel_Rot_Cython(np.deg2rad(array[0]),np.deg2rad(array[1]),np.deg2rad(0),np.swapaxes(pol_cor,axis1 = 1,\
                axis2 = 0),order = 30,sigma=0.5*0.25**2,Number_Grid = pol_cor.shape[1])
    Cone        = Radial_Part*Get_Kernel_Rot
    Cones.append(Cone)
    
    print(f'Finished -----"{k}"----- Mask  for Anlge ---"{array}"--- in --"{time.time()-Time_Start}"-- seconds')

    k += 1 

' Save_Filter_Bank '

np.save(os.path.join("Filter_Mask_Orientation_Score_3D/","Wavelet_Filter_new"),np.array(Cones))