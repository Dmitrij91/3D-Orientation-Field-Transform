import numpy as np
import os
import sys
sys.path.append('../')
from Line_Filter_Transform import Euler_Angles_Sphere
import argparse 
from Generate_Vessel import Synthatic_Data_Random_Walk

parser = argparse.ArgumentParser(description='Synthatic_Data_Set_with_Bifurcations')

parser.add_argument("Bif_Number",
    type=int,
    help="Generated_Bifurcations",
    default = 100)
parser.add_argument("D_33",
    type=np.double,
    help="Diffusion_Coef_for_spatial_regularization",
    default =1)
parser.add_argument("D_44",
    type=np.double,
    help="Diffusion_Coef_for_angular_regularization",
    default = 1)
parser.add_argument("Vessel_Length",
    type=int,
    help="Length_of_each_Vessel",
    default = 15)
parser.add_argument("mean",
    type= np.double,
    help="mean_multiplicative_noise",
    default = 0)
parser.add_argument("std",
    type= np.double,
    help="standard_deviation",
    default = 3)
args = parser.parse_args()

Init_Or  = np.array([1,1,0.1])/np.linalg.norm(np.array([1,1,0.1])).astype(np.float64)
Volume,Volume_noisy,Test_Syn,Test_Rad = Synthatic_Data_Random_Walk(args.Bif_Number,Init_Or,1,1,args.Vessel_Length,args.mean,args.std)


' Save_Volume '

Path = '../Data_Folder/Synthatic_Data_Sets/Synthatic_Vol_'+str(args.Bif_Number)

if not os.path.exists('../Data_Folder/Synthatic_Data_Sets'):
    os.mkdir('../Data_Folder/Synthatic_Data_Sets')

if not os.path.exists(Path):
    os.mkdir(Path)


np.save(os.path.join(Path,'Volume_Syn_Ground_Truth'+str(args.Bif_Number)),Volume)

np.save(os.path.join(Path,'Volume_Syn_Noisy'+str(args.Bif_Number)),Volume_noisy)
np.save(os.path.join(Path,'Vessel_Centerline_'+str(args.Bif_Number)),Test_Syn)
np.save(os.path.join(Path,'Vessel_Radius_Filled_'+str(args.Bif_Number)),Test_Rad)