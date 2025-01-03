
import numpy as np
cimport numpy as cnp
import cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport signbit,exp, pi, fmax,sqrt,abs
import time 

cdef extern from "math.h"nogil:
    
    cdef double pow_double "pow"(double,double)
    cdef double f_min_double "fmin"(double,double)
    cdef double f_max_double "fmax"(double,double)
    
cdef extern from "math.h":
    int pow(int,int) nogil
    int fmin(int,int) nogil
    
from scipy.linalg.cython_lapack cimport dgeqrf
from scipy.linalg.cython_blas cimport dger
from scipy.linalg.cython_blas cimport ddot
from openmp cimport omp_get_max_threads, omp_get_thread_num

cdef extern from "math.h"nogil:
    
    cdef double fmod(double,double)
    cdef double atan2(double,double)
    cdef double acos(double)
    cdef double sin(double)
    cdef double cos(double)
    cdef int round(double)
    
ctypedef fused TYPE:
    double
    int
' Energy functional for Fast Marching as proposed by V. Mohan et al  '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef Energy_Radius(double [:,:,:,::1] Data, double [:] Rad_Array,double [:,:] Orient_Array,double Rad_min,double Rad_max,int Rad_num,int Theta_num):
    
    cdef int X_dim = Data.shape[0]
    
    cdef int Y_dim = Data.shape[1]
    
    cdef int Z_dim = Data.shape[2]
    
    cdef int Or_dim = Data.shape[3]
    
    cdef double alpha = 1.5
    
    cdef int Rad_dim = Rad_Array.shape[0]
    
    ' Returned Potential with small values on the centerline within tubular structures '
    
    cdef double [:,:,:,:,::1] Potential_Ret = np.zeros((X_dim,Y_dim,Z_dim,Or_dim,Rad_dim)) 
    
    cdef int s,k,l,j,i,h
    
    cdef int tid, num_threads = omp_get_max_threads()
    
    cdef int* Node_Coord = <int*>malloc(sizeof(int)*3*X_dim*Y_dim*Z_dim)

    cdef double* Func_mean_disk = <double*>malloc(sizeof(double)*X_dim*Y_dim*Z_dim*Or_dim*Or_dim)

    cdef double* Func_mean_annular = <double*>malloc(sizeof(double)*X_dim*Y_dim*Z_dim*Or_dim*Or_dim)

    for k in prange(X_dim*Y_dim*Z_dim*Or_dim*Or_dim,nogil = True):

        Func_mean_disk[k] = 0

        Func_mean_annular[k] = 0

    cdef double time_start = time.time()

    cdef int X_id, Y_id, Z_id

    for k in prange(X_dim*Y_dim*Z_dim,nogil = True):

        ' Convert Loop id to grid indices '

        X_id = k/(Y_dim*Z_dim)

        Y_id = (k%(Y_dim*Z_dim))/Z_dim

        Z_id = (k%(Y_dim*Z_dim))%Z_dim

        Node_Coord[3*X_id*Y_dim*Z_dim+3*Y_id*Z_dim+3*Z_id] = X_id
                        
        Node_Coord[3*X_id*Y_dim*Z_dim+3*Y_id*Z_dim+3*Z_id+1] = Y_id
                    
        Node_Coord[3*X_id*Y_dim*Z_dim+3*Y_id*Z_dim+3*Z_id+2] = Z_id

        for j in range(Or_dim):
            
            for s in range(Rad_dim):
                
                Get_Mean_Diff_Disk(Data,Rad_Array[s],Orient_Array[j,:]\
                                                    ,Orient_Array,X_dim,&Node_Coord[3*X_id*Y_dim*Z_dim+3*Y_id*Z_dim+3*Z_id],\
                                                        &Func_mean_disk[X_id*Y_dim*Z_dim*Or_dim*Or_dim+Y_id*Z_dim*Or_dim*Or_dim+Z_id*Or_dim*Or_dim+j*Or_dim],Rad_min,Rad_num,Theta_num)
                
                Get_Mean_Diff_Annular(Data,Rad_Array[s],Orient_Array[j,:]\
                                                    ,Orient_Array,X_dim,alpha,&Node_Coord[3*X_id*Y_dim*Z_dim+3*Y_id*Z_dim+3*Z_id],\
                                                        &Func_mean_annular[X_id*Y_dim*Z_dim*Or_dim*Or_dim+Y_id*Z_dim*Or_dim*Or_dim+Z_id*Or_dim*Or_dim+j*Or_dim],Rad_min,Rad_num,Theta_num)
                
                
                
                Potential_Ret[X_id,Y_id,Z_id,j,s] = 1.0/(1+(Spherical_Norm(&Func_mean_disk[X_id*Y_dim*Z_dim*Or_dim*Or_dim+Y_id*Z_dim*Or_dim*Or_dim+Z_id*Or_dim*Or_dim+j*Or_dim],\
                                                &Func_mean_annular[X_id*Y_dim*Z_dim*Or_dim*Or_dim+Y_id*Z_dim*Or_dim*Or_dim+Z_id*Or_dim*Or_dim+j*Or_dim],Or_dim,Orient_Array)))
                
                        
    free(Node_Coord)
    free(Func_mean_disk)
    free(Func_mean_annular)
    print('Computation finished in ___'+str(int((time.time()-time_start)/60))+'___minutes'+'___'+str(int((time.time()-time_start)%60))+'___seconds')

    return Potential_Ret

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef void Get_Mean_Diff_Disk(double [:,:,:,::1] Data,double radius,double [:] Input_or,\
                                double [:,:] Or_Array,int sp_dim,int* Node_Coord_1,double* Mean_Profile,double Rad_min,int Rad_num,int Num_Theta)nogil:
    
    cdef int dim_or = Or_Array.shape[0]
    
    cdef int k,l,r,d,n,m
    
    cdef int X = sp_dim
    
    cdef int Y = sp_dim
     
    cdef int Z = sp_dim
    
    cdef double* Orth_Vec_Array = <double*>malloc(sizeof(double)*Num_Theta*3) 
    
    cdef double* Orth_Vec_pt 
    
    cdef double* Array_Theta_pt   
    
    for k in range(Num_Theta):
        
        Orth_Plane(Input_or,k*(2*pi/(Num_Theta-1)),&Orth_Vec_Array[k*3])

    cdef int* Node_Coord 
    
    cdef double Theta_Angle

    cdef double Radius

    for m in prange(dim_or):

        for r in range(Rad_num):
                        
            for d in range(Num_Theta):
                            
                Node_Coord_pt  = &Node_Coord_1[0]
                
                Theta_Angle = d*(2*pi/(Num_Theta-1))
                            
                Radius      = Rad_min+(r*radius)/(Rad_num-1) 
                            
                Orth_Vec_pt    = &Orth_Vec_Array[d*3]
                            
                Node_Coord = Spatial_Coordinates_Disk(Node_Coord_pt,&Radius,&Theta_Angle,Orth_Vec_pt,X)
                
                Mean_Profile[m] += 1.0/(radius*radius)*Data[Node_Coord[0],Node_Coord[1]\
                                    ,Node_Coord[2],m]*Radius*(1.0/Rad_num)\
                                                        *(2*pi/Num_Theta) 
                free(Node_Coord)
    
    free(Orth_Vec_Array)
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef int* Spatial_Coordinates_Disk(int* Node_Coord,double* radius, double* angle,double* Orth_Vec,\
                                           int boundary_pos)nogil:
    
    cdef int* Vec_Tr_pos        
    
    cdef int* Node_Coord_new = <int*>malloc(sizeof(int)*3)
    
    cdef int* Vec_Tr_pos_pt
    
    ' Get Coordinates of the translated vector of input Radius '
    
    #print(f'--"{Node_Coord[0]}""--')
    
    Vec_Tr_pos = Get_Spatial_Pos(Node_Coord,radius[0],Orth_Vec)
    
    cdef int k,l
    
    for k in range(3):
        
        Node_Coord_new[k] = Node_Coord[k]
    
    cdef int min_check
    
    cdef int max_check
    
    min_check = Vec_Tr_pos[0] 
    
    max_check = Vec_Tr_pos[0] 
    
    for k in range(1,3):
        
        if min_check > Vec_Tr_pos[k]:
            
            min_check = Vec_Tr_pos[k]
            
        if max_check < Vec_Tr_pos[k]:
            
            max_check = Vec_Tr_pos[k]
    
    
    if max_check <= boundary_pos-1 and min_check >= 0:
        
        Node_Coord_new[0] = Vec_Tr_pos[0]
        
        Node_Coord_new[1] = Vec_Tr_pos[1]
        
        Node_Coord_new[2] = Vec_Tr_pos[2]
 
        
    free(Vec_Tr_pos)
    
    return &Node_Coord_new[0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef void Get_Mean_Diff_Annular(double [:,:,:,::1] Data,double radius,double [:] Input_or,\
                                double [:,:] Or_Array,int sp_dim,double alpha,int* Node_Coord_1,double* Mean_Profile_Annular,double Rad_min,int Rad_num,int Num_Theta)nogil:
    
    cdef int dim_or = Or_Array.shape[0]
    
    cdef int k,l,r,d,n,m
    
    cdef int X = sp_dim
    
    cdef int Y = sp_dim
     
    cdef int Z = sp_dim
    
    cdef double* Orth_Vec_Array = <double*>malloc(sizeof(double)*Num_Theta*3) 
    
    cdef double* Orth_Vec_pt 
    
    ' Get Radii for the annular region '  
    
    for k in range(Num_Theta):
        
         Orth_Plane(Input_or,k*(2*pi/(Num_Theta-1)),&Orth_Vec_Array[k*3])
     
    cdef int* Node_Coord 

    cdef double Theta_Angle

    cdef double Radius

    for m in prange(dim_or):

        for r in range(Rad_num):
                        
            for d in range(Num_Theta):
                            
                Node_Coord_pt  = &Node_Coord_1[0]
                
                Theta_Angle = d*(2*pi/(Num_Theta-1))
                            
                Radius      = Rad_min+radius+(k*(radius*alpha-radius))/(Rad_num-1)
                            
                Orth_Vec_pt    = &Orth_Vec_Array[d*3]
                            
                Node_Coord = Spatial_Coordinates_Disk(Node_Coord_pt,&Radius,&Theta_Angle,Orth_Vec_pt,X)
                            
                Mean_Profile_Annular[m] += 1.0/((alpha*alpha-1)*radius*radius)*\
                                                Data[Node_Coord[0],Node_Coord[1]\
                                                ,Node_Coord[2],m]*Radius*(1.0/Rad_num)\
                                                *(2*pi/Num_Theta)
                free(Node_Coord)
    

    free(Orth_Vec_Array)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef double Spherical_Norm(double* func_1,double* func_2,int dim,double [:,:] Or_Array)nogil:
    
    cdef int k
    
    cdef double Theta = 0
    
    cdef double Phi = 0  
    
    cdef double Integral = 0
    
    for k in range(dim):
        
        Theta = acos(Or_Array[k,2])
    
        #Phi   = fmod(atan2(Or_Array[k,1],Or_Array[k,0]),2*pi) 
    
        Integral += (func_1[k]-func_2[k])*(func_1[k]-func_2[k])*sin(Theta)/dim
    
    return Integral


' Get nearest spatial positition '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef int* Get_Spatial_Pos(TYPE* Ref_Coord, double Radius,double* Direction) nogil:
    
    cdef int* Vec_tr = <int*>malloc(sizeof(int)*3)
    
    cdef int k 
    
    for k in range(3):
        
        Vec_tr[k] = round(Ref_Coord[k]+Radius*Direction[k])
    
    return &Vec_tr[0]

' Return Orthogonal unit vertor in direction alpha within the orthogonal plane to input direction  ' 
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


cdef void Orth_Plane(double[:] direction,double alpha,double* orth_vec_new)nogil:
    
    
    ' Intitialization '
    
    cdef int i,j 

    cdef double* Q = <double*>malloc(sizeof(double)*3*3)

    for i in range(3*3):

        Q[i] = 0

    for j in range(3):

        Q[j*3+j] = 1


    cdef double* outer        = <double*>malloc(sizeof(double)*3)
    
    cdef double tau  = 0
    
    cdef double work = 0
    
    cdef int lwork = 1
    
    cdef int info = 0
    
    cdef int dim_m = 3
    
    cdef int dim_n = 1
    
    cdef int k
    
    for k in range(3):
        
        outer[k] = direction[k]
    
    dgeqrf(&dim_m,&dim_n,outer,&dim_m,&tau,&work,&lwork,&info)
    
    tau = - tau
    
    outer[0] = 1
    
    dger(&dim_m,&dim_m,&tau,outer,&dim_n,outer,&dim_n,&Q[0],&dim_m)
    
    ' Overwrite to oriented vector '
    
    for k in range(3):
        
        orth_vec_new[k] = cos(alpha)*Q[k*3+1]+sin(alpha)*Q[k*3+2]
        
        
    free(outer)
    free(Q)

    ' Add for Line Profiler '

def Energy_Radius_Dummy():
    pass