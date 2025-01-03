import os.path
import logging
import numpy as np
import cython
cimport numpy as cnp
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from openmp cimport omp_get_max_threads, omp_get_thread_num
from libc.math cimport signbit,exp, pi,pow,sqrt
from cython.view cimport array as cvarray
from libc.string cimport memcpy
from Fast_Marching_Cython.Stochastic_Kernel_Cython import Random_walk_Kernel_Cython
cimport scipy.linalg.cython_lapack as lapack
from scipy.linalg.cython_blas cimport dger,ddot

' Function to return kernel value on Orientation Score '

###############################################################################
#      Implementation of paper : Improving Fiber Alignment in HARDI by   ######
#      Combining Contextual PDE Flow with                                ######
#      Constrained Spherical Deconvolution                               ######
#      published by Remco research Group: J. M. Portegies                ######
############################################################################### 

cdef extern from "math.h":
    cdef int ceil(double) nogil
    cdef double abs(double) nogil
    cdef double tan(double x) nogil
    cdef double sin(double x) nogil
    cdef double sinh(double) nogil
    cdef double cos(double x) nogil
    cdef double exp(double) nogil
    cdef double sqrt(double) nogil
    cdef double atan2(double x,double y) nogil
    cdef double acos(double x) nogil
'  Sort Index List without output '   
    
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

cdef struct IndexedElement:
    cnp.ulong_t index
    cnp.float64_t value

cdef int _compare(const_void *a, const_void *b):
    cdef cnp.float64_t v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
    if v < 0: return -1
    if v >= 0: return 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef void argsort(double [:] data, int [:] order):
    cdef cnp.ulong_t i
    cdef cnp.ulong_t n = data.shape[0]
    
    # Allocate index tracking array.
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement))
    
    # Copy data into index tracking array.
    for i in range(n):
        order_struct[i].index = i
        order_struct[i].value = data[i]
        
    # Sort index tracking array.
    qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)
    
    # Copy indices from index tracking array to output array.
    for i in range(n):
        order[i] = order_struct[i].index
        
    # Free index tracking array.

    free(order_struct)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
    
cdef void argsort_carray(double* data, int* order,int Length) nogil:
    cdef cnp.ulong_t i
    cdef cnp.ulong_t n = Length
    
    # Allocate index tracking array.
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement))
    
    # Copy data into index tracking array.
    for i in range(n):
        order_struct[i].index = i
        order_struct[i].value = data[i]
        
    # Sort index tracking array.
    qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)
    
    # Copy indices from index tracking array to output array.
    for i in range(n):
        order[i] = order_struct[i].index
        
    # Free index tracking array.

    free(order_struct)

cdef extern from 'complex.h' nogil:
    double creal(complex)
    double complex I
    double cabs(complex)
    double carg(complex)
    double complex cpow(complex,int)    
    
cdef extern from "math.h" nogil:
    int fmax(int,int)
    int fmin(int,int)
    
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)

cdef double kernel(double x,double y,double z,double beta,double gamma,double D_33,\
                   double D_44,double time_state) nogil:

    cdef double Value = (8.0/sqrt(2))*sqrt(pi*D_44)*D_33*time_state
    
    cdef double Kernel_1_D_Response_beta  = Kernel_1_D(z/2,x,beta,time_state,D_33,D_44)

    cdef double Kernel_1_D_Response_gamma = Kernel_1_D(z/2,-y,gamma,time_state,D_33,D_44)
    
    Value = Value * Kernel_1_D_Response_beta * Kernel_1_D_Response_gamma
    
    return Value


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)

cdef double Gaussian(double norm_square,double time) nogil:

    return  1.0/(pow(4*pi*time,1.5))*exp(-norm_square/(4*time))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)

cdef double Orient_dis(double* Orient_x, double* Orient_y, double kappa) nogil:

    
    cdef double value = 0

    cdef int* dim         = < int*>malloc(sizeof(int))
    
    ' Assign strides for inner product '
    
    dim[0]        = 3
    
    cdef double* ptr_or_1 = Orient_x
    
    cdef double* ptr_or_2 = Orient_y

    cdef int* stride_or   = < int*>malloc(sizeof(int)) 
    
    stride_or[0]  = 1
    
    value = ddot(dim,ptr_or_1,stride_or,ptr_or_2,stride_or)

    free(dim)
    free(stride_or)

    return  kappa*exp(kappa*value)/(4*pi*sinh(kappa))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)

cdef double Kernel_Mises_Fisher(double* Sp_pos_x, double* Sp_pos_y, double* Orient_x,double* Orient_y,double time_state,double kappa) nogil:

    cdef double Value = 0
    
    cdef double* Diff = <double*>malloc(sizeof(double)*3)

    cdef double Norm

    cdef double Kernel_dist, Kernel_orient , Kernel_fiber

    cdef double* ptr_or_x
    
    cdef double* ptr_or_y
    
    cdef double* ptr_pos_x
    cdef double* ptr_pos_y

    cdef int k

    for k in range(3):

        Norm += (Sp_pos_x[k]-Sp_pos_y[k])*(Sp_pos_x[k]-Sp_pos_y[k])

    for k in range(3):

        Diff[k] = (Sp_pos_y[k]-Sp_pos_x[k])/sqrt(Norm)

    ptr_pos_x = &Sp_pos_x[0]

    ptr_pos_y = &Sp_pos_y[0]

    ptr_or_x = &Orient_x[0]

    ptr_or_y = &Orient_y[0]

    Kernel_dist = Gaussian(Norm,time_state)

    Kernel_orient = Orient_dis(ptr_or_x,ptr_or_y,kappa)

    Kernel_fiber = Orient_dis(&Diff[0],ptr_or_x,kappa)

    Value =  Kernel_dist * Kernel_orient* Kernel_fiber 
    
    free(Diff)

    return Value/(4*pi)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)

cdef double Kernel_1_D(double c_1,double c_2, double c_3,double time_state,double D_33,double D_44) nogil:
    
    cdef double Exponent = Get_Exponent(c_1,c_2,c_3,D_33,D_44)
    
    cdef double Value = (1.0/(32*pi*pow(time_state,2)*D_33*D_44))*exp(-sqrt(Exponent/(4*time_state)))
    
    return Value
                           
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)                           

cdef double Get_Exponent(double x_pos,double y_pos, double angle,double D_33,double D_44) nogil:
    
    cdef double var = angle/(2*tan(angle/2))
    
    if abs(angle) < pi/10:
        
        var = cos(angle/2)/(1-(pow(angle,2))/24)
    
    cdef double exponent = pow((pow(angle,2)/D_44+(1/D_33)*pow((angle*y_pos/2+var\
                        *x_pos),2)),2)+1.0/(D_33*D_44)*pow((-x_pos*angle/2+var*y_pos),2)
    
    
    return exponent 

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)                           

cdef Get_Aligned_Kernel(double [:] sp_pos_x,double [:] angle_or_x,double [:] sp_pos_y,double [:] angle_or_y\
                       ,double D_33,double D_44,double time_step,str Method = 'Mises_Fischer_Kernel'):
    

    cdef double Kernel_Value = 0
    
    cdef double beta_rotated = 0
    
    cdef double gamma_rotated = 0

    cdef double kappa = 4

    cdef int k
    
    ' Define x-y '
    
    cdef int dim = sp_pos_x.shape[0]
    
    cdef double [:] Diff_Vec_pos = np.zeros(dim)
    
    for k in range(dim):
        
        Diff_Vec_pos[k] = sp_pos_x[k] - sp_pos_y[k]
        
    cdef double [:,:] Rotation_Matrix = Get_Rot_Mat_Euler_Angles(0,angle_or_y[0],angle_or_y[1])
    
    cdef double [:] n_vec = np.zeros(3)
    
    cdef double [:] n_vec_1 = np.zeros(3)
    
    n_vec_x = Euler_to_Orientation(angle_or_x[0],angle_or_x[1])
    
    n_vec_y = Euler_to_Orientation(angle_or_y[0],angle_or_y[1])
    
    cdef double [:] orientation_1 = np.zeros(dim)
    
    cdef double [:] orientation_2 = np.zeros(dim)
    
    ' Rotate Orientations '
    
    for k in range(dim):
    
        orientation_1[0] += Rotation_Matrix[0,k]*Diff_Vec_pos[k] 

        orientation_1[1] += Rotation_Matrix[1,k]*Diff_Vec_pos[k]

        orientation_1[2] += Rotation_Matrix[2,k]*Diff_Vec_pos[k]

        orientation_2[0] += Rotation_Matrix[0,k]*n_vec_x[k]

        orientation_2[1] += Rotation_Matrix[1,k]*n_vec_x[k]

        orientation_2[2] += Rotation_Matrix[2,k]*n_vec_x[k]
    
    
    beta_rotated,gamma_rotated = Orientation_to_Euler(orientation_2)

    if Method == 'Kernel_2D':

        Kernel_Value = kernel(orientation_1[0],orientation_1[1],orientation_1[2],\
                          beta_rotated,gamma_rotated,D_33,D_44,time_step)

    elif Method == 'Mises_Fischer_Kernel':

        Kernel_Value = Kernel_Mises_Fisher(&sp_pos_x[0],&sp_pos_y[0],&orientation_1[0],&orientation_2[0],time_step, kappa)     
    
    return Kernel_Value

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)


cdef void Euler2Quaternion_vector(double alpha,double beta,double gamma, double* quaternion):
    
    cdef double ca = cos(alpha * 0.5)
    cdef double sa = sin(alpha * 0.5)
    cdef double cb = cos(beta * 0.5)
    cdef double sb = sin(beta * 0.5)
    cdef double cc = cos(gamma * 0.5)
    cdef double sc = sin(gamma * 0.5)
    
    
    

    quaternion[0] = ca*cb*cc-sa*cb*sc
    quaternion[1] = sa*cb*cc+ca*cb*sc
    quaternion[2] = ca*sb*sc-sa*sb*cc
    quaternion[3] = ca*sb*cc+sa*sb*sc
    
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True) 

cdef double[:,:] Get_Rot_Mat_Euler_Angles(double alpha,double beta,double gamma):
    
    cdef double* quaternion = <double*>malloc(sizeof(double)*4)
    
    cdef int k 
    
    for k in range(4):
        
        quaternion[k] = 0
    
    Euler2Quaternion_vector(alpha,beta,gamma,&quaternion[0])
    
    cdef double [:,:] Rot_Matrix = np.zeros((3,3))
    
    cdef double [:] n_vec = np.zeros(3)
    
    ' Convert to transposed rotation matrix : if not unit vector --> scalar multiple of Orthogal matrix '
    
    Rot_Matrix[0,0] = quaternion[0]**2+quaternion[1]**2-quaternion[2]**2-quaternion[3]**2
    Rot_Matrix[1,0] = 2*(quaternion[1]*quaternion[2]-quaternion[0]*quaternion[3])
    Rot_Matrix[2,0] = 2*(quaternion[0]*quaternion[2]+quaternion[1]*quaternion[3])
    Rot_Matrix[0,1] = 2*(quaternion[1]*quaternion[2]+quaternion[0]*quaternion[3])
    Rot_Matrix[1,1] = quaternion[0]**2-quaternion[1]**2+quaternion[2]**2-quaternion[3]**2
    Rot_Matrix[2,1] = 2*(quaternion[2]*quaternion[3]-quaternion[0]*quaternion[1])
    Rot_Matrix[0,2] = 2*(quaternion[1]*quaternion[3]-quaternion[0]*quaternion[2])
    Rot_Matrix[1,2] = 2*(quaternion[0]*quaternion[1]+quaternion[2]*quaternion[3])
    Rot_Matrix[2,2] = quaternion[0]**2-quaternion[1]**2+quaternion[2]**2-quaternion[3]**2
    
    free(quaternion)
    
    return Rot_Matrix

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)  

cdef double [:] Euler_to_Orientation(double beta,double gamma):
    
    cdef double [:] orientation = np.zeros(3)
    
    orientation[0] = sin(beta)
    
    orientation[1] = -sin(gamma)*cos(beta)
    
    orientation[2] = cos(beta)*cos(gamma)
    
    return orientation


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)  

cdef inline (double,double) Orientation_to_Euler(double [:] orientation):

    cdef double beta  = 0
    
    cdef double gamma = 0
    
    if orientation[0]**2 < 10e-6 and orientation[1]**2 < 10e-6 and (orientation[2]-1)**2 < 10e-6:
        
        beta  = 0
        
        gamma = 0
        
    elif orientation[0]**2 < 10e-6 and orientation[1]**2 < 10e-6 and (orientation[2]+1)**2 < 10e-6:
        
        beta = pi
        
        gamma = 0
        
    else:
        
        beta = acos(orientation[2])
        
        gamma = atan2(orientation[1], orientation[0])
        
    return beta,gamma

' Get nearest angle on the discrete grid '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef int* Get_Discrete_Dir(double [:] Input,double [:,:] Dir_Array, int Num_angle):

    cdef int k
    
    cdef int* dim         = < int*>malloc(sizeof(int))
    
    ' Assign strides for inner product '
    
    dim[0]        = 3
    
    cdef double* ptr_or
    
    cdef int* stride_or   = < int*>malloc(sizeof(int)) 
    
    stride_or[0]  = 1

    
    cdef int Length = Dir_Array.shape[0]
    
    cdef int* Return_ind = <int*>malloc(sizeof(int)*Num_angle)
    
    cdef double beta,gamma,beta_1,gamma_1
    
    cdef double* Distance = <double*>malloc(sizeof(double)*Length)
    
    cdef int* index_order =<int*>malloc(sizeof(int)*Length) 
    

    for k in range(Length):
        
        index_order[k] = k

        ptr_or      = &Dir_Array[k,:][0]
        
        Distance[k] = acos(ddot(dim,&Input[0],stride_or,ptr_or,stride_or))

    argsort_carray(Distance,&index_order[0],Length)
    
    for k in range(Num_angle):

        Return_ind[k] = index_order[k]
    
    free(Distance)
    free(index_order)
    free(stride_or)
    free(dim)
    
    return &Return_ind[0]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef int Get_Discrete_Dir_Stochastic(double [:] Input,double [:,:] Dir_Array):

    cdef int k
    
    cdef int* dim         = < int*>malloc(sizeof(int))
    
    ' Assign strides for inner product '
    
    dim[0]        = 3
    
    cdef double* ptr_or
    
    cdef int* stride_or   = < int*>malloc(sizeof(int)) 
    
    stride_or[0]  = 1

    
    cdef int Length = Dir_Array.shape[0]
    
    cdef int Return_ind 
    
    cdef double* Distance = <double*>malloc(sizeof(double)*Length)
    
    cdef int* index_order =<int*>malloc(sizeof(int)*Length) 
    

    for k in range(Length):
        
        index_order[k] = k

        ptr_or      = &Dir_Array[k,:][0]
        
        Distance[k] = acos(ddot(dim,&Input[0],stride_or,ptr_or,stride_or))

    argsort_carray(Distance,&index_order[0],Length)
    
    Return_ind = index_order[0]

    free(Distance)
    free(index_order)
    free(stride_or)
    free(dim)
    
    return Return_ind

' Implementation of Convolution on Orientation Score Domain '

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)  

cpdef double [:,:,:,:] convolution_routine(double [:,:,:,:] data, double D_33,double D_44,double time_step,\
                                          int kernel_size,double [:,::1] orientation_list,double [:,::1] angle_list,int num_angl_conv,str Method = 'Kernel_2D'):
    
    ' Initilize Variables for the convolution routine '
    
    cdef int dim_x  = data.shape[0]
    
    cdef int dim_y  = data.shape[1]
    
    cdef int dim_z  = data.shape[2]
    
    cdef int dim_or = data.shape[3]
    
    cdef int dim_or_kernel = num_angl_conv
    
    ' Set window size '
    
    cdef int dim_conv_xyz = (kernel_size-1)/2
    
    cdef int or_data,x_data,y_data,z_data,x_kernel,y_kernel,z_kernel,or_kernel
    
    cdef double [:,:,:,:] Data_Convolved = np.zeros((dim_x,dim_y,dim_z,dim_or))
    
    cdef double [:,:,:,:] Number_It = np.zeros((dim_x,dim_y,dim_z,dim_or))
    
    cdef double [:,:,:,:] Data_Convolved_loop = np.zeros((dim_x,dim_y,dim_z,dim_or))
    
    
    ' Create Convolution Kernel centered at origin '
    
    cdef double [:,:,:,:,:] conv_kernel = np.zeros((kernel_size,kernel_size,kernel_size,\
                                                    dim_or_kernel,dim_or))
    
    ' Variables for spatial kernel coordinates around origin '
    
    cdef double [:] x_sp = np.zeros(3)
    
    cdef double [:] y_sp = np.zeros(3)
    
    cdef int angle_1,angle_2,x_it,y_it, z_it
    
    cdef int* Or_Ind
    cdef int l

    for angle_1 in range(dim_or):
        
        Or_Ind = Get_Discrete_Dir(orientation_list[angle_1],orientation_list,num_angl_conv)

        for angle_2 in range(dim_or_kernel):
            
            for x_it in range(-dim_conv_xyz,dim_conv_xyz+1):
    
                for y_it in range(-dim_conv_xyz,dim_conv_xyz+1):
            
                    for z_it in range(-dim_conv_xyz,dim_conv_xyz+1): 
                    
                        x_sp[0] = x_it 
                        
                        x_sp[1] = y_it
    
                        x_sp[2] = z_it
                    
                        conv_kernel[x_it+dim_conv_xyz,y_it+dim_conv_xyz,z_it+dim_conv_xyz,angle_2,angle_1] = \
                           Get_Aligned_Kernel(x_sp,angle_list[angle_1],y_sp,angle_list[Or_Ind[angle_2]]\
                                              ,D_33,D_44,time_step,Method)
                        
        free(Or_Ind)

    ' Perform Convolution Main Routine '

    for or_data in prange(dim_or,nogil= True):
        
        for x_data in range(dim_x):
            
            for y_data in range(dim_y):
                
                for z_data in range(dim_z):
                    
                    ' Up here iterate over the kernel values '
                    
                    for x_kernel in range(fmax(x_data-dim_conv_xyz,0),\
                                          fmin(x_data+dim_conv_xyz+1,dim_x-1)):
                        
                        for y_kernel in range(fmax(y_data-dim_conv_xyz,0),\
                                          fmin(y_data+dim_conv_xyz+1,dim_y-1)):
                            
                            for z_kernel in range(fmax(z_data-dim_conv_xyz,0),\
                                          fmin(z_data+dim_conv_xyz+1,dim_z-1)):
                                
                                Number_It[x_data,y_data,z_data,or_data] += 1.0
                                
                                for or_kernel in range(0,dim_or_kernel):
                                    # Set indices within kernel to start by zero
                                    Data_Convolved_loop[x_data,y_data,z_data,or_data] += data[x_kernel,\
                                    y_kernel,z_kernel,or_data]*conv_kernel[-x_data+dim_conv_xyz+x_kernel,\
                                                        -y_data+dim_conv_xyz+y_kernel,\
                                                        -z_data+dim_conv_xyz+z_kernel,or_kernel,or_data]
                    
                    ' Assign convolved data to output variable '
                    
                    Data_Convolved[x_data,y_data,z_data,or_data] = Data_Convolved_loop[x_data,y_data,z_data,\
                                                                                 or_data]*dim_x*dim_y*dim_z\
                                                                     /Number_It[x_data,y_data,z_data,or_data] 
                    
    return Data_Convolved
                            
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)


cdef void Rotate_kernel_stochastic(double* Sp_pos, int* index_rot, double[:,:] orientation_list,int num_angl_conv):

    cdef double alpha, beta, gamma 

    cdef int k,l

    cdef double [:] Or_Array = orientation_list[index_rot[0],:]
 
    beta,gamma = Orientation_to_Euler(Or_Array) 

    cdef double [:,:] Orient_rot  

    cdef double [:] Or_Array_Ret = np.zeros(3)

    alpha = 0 

    Rot_Mat = Get_Rot_Mat_Euler_Angles(alpha,beta,gamma)

    for k in range(3):

        Sp_pos[k] = round(Rot_Mat[0,k]*Sp_pos[0] +Rot_Mat[1,k]*Sp_pos[1]+Rot_Mat[2,k]*Sp_pos[2])

        Or_Array_Ret[k] = Rot_Mat[0,k]*Or_Array[0] +Rot_Mat[1,k]*Or_Array[1]+Rot_Mat[2,k]*Or_Array[2]
    

    index_rot[0] = Get_Discrete_Dir_Stochastic(Or_Array_Ret,orientation_list)
    

' Convolution routine with stochastic kernel '

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)  

cpdef double [:,:,:,:] convolution_routine_stochastic(double [:,:,:,:] data, double D_33,double D_44,double time_step,\
                                          int kernel_size,double [:,::1] orientation_list,double [:,::1] angle_list,\
                                              int num_angl_conv,str Method = 'Contour_Enh'):
    
    ' Initilize Variables for the convolution routine '
    

    cdef int dim_x  = data.shape[0]
    
    cdef int dim_y  = data.shape[1]
    
    cdef int dim_z  = data.shape[2]
    
    cdef int dim_or = data.shape[3]
    
    cdef int dim_or_kernel = num_angl_conv
    
    ' Set window size '
    
    cdef int dim_conv_xyz = (kernel_size-1)/2
    
    cdef int or_data,x_data,y_data,z_data,x_kernel,y_kernel,z_kernel,or_kernel
    
    cdef double [:,:,:,:] Data_Convolved = np.zeros((dim_x,dim_y,dim_z,dim_or))
    
    cdef double [:,:,:,:] Number_It = np.zeros((dim_x,dim_y,dim_z,dim_or))
    
    cdef double [:,:,:,:] Data_Convolved_loop = np.zeros((dim_x,dim_y,dim_z,dim_or))
    
    
    ' Create Convolution Kernel centered at origin '
    
    cdef double [:,:,:,:,:] conv_kernel = np.zeros((kernel_size,kernel_size,kernel_size,\
                                                    dim_or_kernel,dim_or))
    
    ' Variables for spatial kernel coordinates around origin '
    
    cdef double [:] x_sp = np.zeros(3)
    
    cdef double [:] y_sp = np.zeros(3)
    
    cdef int angle_1,angle_2,x_it,y_it, z_it
    
    cdef int* Or_Ind
    
    cdef int l

    cdef double [:,:,:,:] Random_walk_kernel

    cdef double [:] init_Or  = np.array([0.1,0.1,1])/np.linalg.norm(np.array([0.1,0.1,1]))
    
    cdef double [:] init_Pos = np.zeros(3)

    cdef int index_rot

    if Method == 'Contour_Enh':

        path = "Stochastic_Kernels"
        
        ' Input: x,y,z,beta,gama '

        if os.path.isfile(os.path.join('Stochastic_Kernels',"Stochastic_Kernel_CE_"+str(time_step)+'_'+str(D_33)+'_'+str(D_44))+str(num_angl_conv)+str(kernel_size)+'.npy'):

            print('Load_Kernel_Cont_Enhancement')

            Random_walk_kernel = np.load('Stochastic_Kernels/Stochastic_Kernel_CE_'+str(time_step)+'_'+str(D_33)+'_'+str(D_44)+str(num_angl_conv)+str(kernel_size)+'.npy')

        else:

            print('Compute_Kernel_Cont_Enhancement')
        

            Random_walk_kernel = Random_walk_Kernel_Cython(init_Pos,init_Or,int(1e8),D_33,D_44,\
                          20,time_step,50,68,orientation_list,Method)
            print('Computation_finished')
            np.save(os.path.join(path,"Stochastic_Kernel_CE_"+str(time_step)+'_'+str(D_33)+'_'+str(D_44)+str(num_angl_conv)+str(kernel_size)),Random_walk_kernel)
    
    elif Method == 'Contour_Compl':
        
        path = "Stochastic_Kernels"

        ' Input: x,y,z,beta,gama '
        
        if os.path.isfile(os.path.join("Stochastic_Kernels","Stochastic_Kernel_CC_"+str(time_step)+'_'+str(D_33)+'_'+str(D_44))+str(num_angl_conv)+str(kernel_size)+'.npy'):

            print('Load_Kernel_Contour_Completion')

            Random_walk_kernel = np.load('Stochastic_Kernels/Stochastic_Kernel_CC_'+str(time_step)+'_'+str(D_33)+'_'+str(D_44)+str(num_angl_conv)+str(kernel_size)+'.npy')

        else:

            print('Compute_Kernel_Contour_Completion')
        

            Random_walk_kernel = Random_walk_Kernel_Cython(init_Pos,init_Or,int(1e8),D_33,D_44,\
                          20,time_step,50,68,orientation_list,Method)

    
            np.save(os.path.join(path,"Stochastic_Kernel_CC_"+str(time_step)+'_'+str(D_33)+'_'+str(D_44)+str(num_angl_conv)+str(kernel_size)),Random_walk_kernel)

    for angle_1 in range(dim_or):
        
        Or_Ind = Get_Discrete_Dir(orientation_list[angle_1],orientation_list,num_angl_conv)

        for angle_2 in range(dim_or_kernel):
                
            index_rot = Or_Ind[angle_2]

            for x_it in range(-dim_conv_xyz,dim_conv_xyz+1):
    
                for y_it in range(-dim_conv_xyz,dim_conv_xyz+1):
            
                    for z_it in range(-dim_conv_xyz,dim_conv_xyz+1): 
                    
                        x_sp[0] = x_it 
                        
                        x_sp[1] = y_it
    
                        x_sp[2] = z_it

                        ' Rotate x_sp and Or_Ind by overwriting the arrays '

                        Rotate_kernel_stochastic(&x_sp[0],&index_rot,orientation_list,num_angl_conv)
        
                        conv_kernel[x_it+dim_conv_xyz,y_it+dim_conv_xyz,z_it+dim_conv_xyz,angle_2,angle_1] = \
                            Random_walk_kernel[25+x_it,25+y_it,25+z_it,index_rot] 
                        #print(str(conv_kernel[x_it+dim_conv_xyz,y_it+dim_conv_xyz,z_it+dim_conv_xyz,angle_2,angle_1])+'Conv_Kernel')
                
        free(Or_Ind)

    ' Perform Convolution Main Routine '

    for or_data in prange(dim_or,nogil = True):
        
        for x_data in range(dim_x):
            
            for y_data in range(dim_y):
                
                for z_data in range(dim_z):
                    
                    Data_Convolved_loop[x_data,y_data,z_data,or_data] = 0

                    Number_It[x_data,y_data,z_data,or_data] = 1

                    ' Up here iterate over the kernel values '
                    
                    for x_kernel in range(fmax(x_data-dim_conv_xyz,0),\
                                          fmin(x_data+dim_conv_xyz+1,dim_x)):
                        
                        for y_kernel in range(fmax(y_data-dim_conv_xyz,0),\
                                          fmin(y_data+dim_conv_xyz+1,dim_y)):
                            
                            for z_kernel in range(fmax(z_data-dim_conv_xyz,0),\
                                          fmin(z_data+dim_conv_xyz+1,dim_z)):
                                
                                Number_It[x_data,y_data,z_data,or_data] += 1.0
                                
                                for or_kernel in range(0,dim_or_kernel):
                                    # Set indices within kernel to start by zero
                                    Data_Convolved_loop[x_data,y_data,z_data,or_data] += data[x_kernel,\
                                    y_kernel,z_kernel,or_data]*conv_kernel[-x_data+dim_conv_xyz+x_kernel,\
                                                        -y_data+dim_conv_xyz+y_kernel,\
                                                        -z_data+dim_conv_xyz+z_kernel,or_kernel,or_data]
                                    #print(str(Data_Convolved_loop[x_data,y_data,z_data,or_data])+'Data_Convolved')
                    
                    ' Assign convolved data to output variable '
                    
                    Data_Convolved[x_data,y_data,z_data,or_data] = Data_Convolved_loop[x_data,y_data,z_data,\
                                                                                 or_data]*dim_x*dim_y*dim_z\
                                                                    /Number_It[x_data,y_data,z_data,or_data] 
                    
    return Data_Convolved

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)

cpdef int Get_Kernel_Size(double D_33, double D_44,double time_step):
       
        
    cdef int kernel_size = 0
    cdef double [:] x
    cdef double [:] y
    cdef double [:] r
    cdef double [:] v
    cdef double i

    x = np.array([0., 0., 0.])
    y = np.array([0., 0., 0.])
    r = np.array([0., 0., 1.])
    v = np.array([0., 0., 1.])

    # evaluate at origin
        
    cdef kernel_max = Get_Aligned_Kernel(x,r,y,v,D_33,D_44,time_step)


    # determine a good kernel size
    i = 0.0
    while True:
        i += 0.1
        x[2] = i
        kval = Get_Aligned_Kernel(x,r,y,v,D_33,D_44,time_step) / kernel_max
        if(kval < 0.1):
            break

    N = ceil(i) * 2
    if N % 2 == 0:
        N -= 1

    kernel_size = N
    
    return kernel_size