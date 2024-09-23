import numpy as np
import math

def Get_Coordinates(I):
    img          = I[:,:,:,None]
    ind          = np.moveaxis(np.indices(img.shape[:-1]),0,-1)
    filter_funcs = []
    
    return ind


def Euler_Angles_Sphere(samples=60):

    points = []
    Euler_Angles = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        ' Get Euler Angles zxz convention x given '
        
        if z > 0:
            
            points.append((x, y, z))
            
            if x**2 < 10e-6 and y**2 < 10e-6 and (z-1)**2 < 10e-6:
        
                beta  = 0
        
                gamma = 0
        
            elif x**2 < 10e-6 and y**2 < 10e-6 and (z+1)**2 < 10e-6:
        
                beta = np.pi
        
                gamma = 0
        
            else:
        
                beta = np.arccos(z)
        
                gamma = np.arctan2(y, x)
#             beta          = np.arccos(z)
            
#             alpha         = np.arcsin(-y/(np.sin(beta)))
        
#             normal_vector = np.cross(np.array([x,y,z]),np.array([1,0,0]))/np.linalg.norm(\
#                                                     np.cross(np.array([x,y,z]),np.array([1,0,0])))

#             gamma         = np.arccos(normal_vector[2]/np.sqrt(1-z**2))
            ###########################################################################
            ####### Comment out first line for Fourier Orientation_Score_filterbank ###
            ####### and the second for the Line Filter Transform                    ###
            ###########################################################################
            
            Euler_Angles.append((np.rad2deg(beta),np.rad2deg(gamma)))
            #Euler_Angles.append((beta,gamma))
        
    return np.array(Euler_Angles), np.array(points)

def Euler_Angles_Sphere_2(samples=60):

    points = []
    Euler_Angles = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        ' Get Euler Angles zxz convention x given '
        
        if z != 0:
            
            points.append((x, y, z))
            
            if x**2 < 10e-6 and y**2 < 10e-6 and (z-1)**2 < 10e-6:
        
                beta  = 0
        
                gamma = 0
        
            elif x**2 < 10e-6 and y**2 < 10e-6 and (z+1)**2 < 10e-6:
        
                beta = np.pi
        
                gamma = 0
        
            else:
        
                beta = np.arccos(z)
        
                gamma = np.arctan2(y, x)
#             beta          = np.arccos(z)
            
#             alpha         = np.arcsin(-y/(np.sin(beta)))
        
#             normal_vector = np.cross(np.array([x,y,z]),np.array([1,0,0]))/np.linalg.norm(\
#                                                     np.cross(np.array([x,y,z]),np.array([1,0,0])))

#             gamma         = np.arccos(normal_vector[2]/np.sqrt(1-z**2))
            ###########################################################################
            ####### Comment out first line for Fourier Orientation_Score_filterbank ###
            ####### and the second for the Line Filter Transform                    ###
            ###########################################################################
            
            Euler_Angles.append((np.rad2deg(beta),np.rad2deg(gamma)))
            #Euler_Angles.append((beta,gamma))
        
    return np.array(Euler_Angles), np.array(points)