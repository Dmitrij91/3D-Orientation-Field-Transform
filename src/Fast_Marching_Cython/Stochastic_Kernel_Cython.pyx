import numpy as np
cimport numpy as cnp
import cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free,rand, RAND_MAX
from openmp cimport omp_get_max_threads, omp_get_thread_num
from libc.math cimport signbit,exp, pi,fabs,asin
from scipy.special.cython_special cimport sph_harm,eval_legendre
from cython.view cimport array as cvarray
from libc.string cimport memcpy
from scipy.linalg.cython_blas cimport ddot, daxpy, dgemm, dgemv

cdef extern from "math.h" nogil:
    double log(double)
    double fmax(double,double)
    double cos(double)
    double sin(double)
    double atan2(double x,double y) nogil
    double acos(double) 
    double sqrt(double) nogil
   # double asin(double)
    double pow(double,int)
    double abs(double) nogil
    int round(double)
    
cdef extern from "math.h"nogil:
    
    cdef double pow_double "pow"(double,double)
    cdef int fmin_int "fmin"(int,int)
    cdef int fmax_int "fmax"(int,int)

' Box Mullar Transform for generating normaly distributed random numbers '

' Uniform Distribution '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef double Rand_Uniform(double mean,double sigma) nogil:
    
    cdef double Random_num = mean+rand()*sigma/(RAND_MAX)
    
    return Random_num

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


cdef double Box_Muller_Normal_Dis(double mean,double sigma) nogil:
    
    cdef double theta
    
    cdef double rad
    
    cdef double X_1,X_2,Z_1
    
    ' Uniform Distribution on [-1,1] '
    
    cdef double w = 2
    
    while w >= 1:
        
        X_1 = 2*Rand_Uniform(0,1)-1
        X_2 = 2*Rand_Uniform(0,1)-1
        w   = X_1*X_1+X_2*X_2
    
    Z_1 = pow_double(-2*log(w)/w,1/2)*X_1

    return mean+sigma*Z_1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


cdef void Assign_Box_Muller(double* Array,double mean,double sigma) nogil:
    
    cdef double theta
    
    cdef double rad
    
    cdef double X_1,X_2,Z_1,Z_2
    
    ' Uniform Distribution on [-1,1] '
    
    cdef double w = 2
    
    while w >= 1:
        
        X_1 = 2*Rand_Uniform(0,1)-1
        X_2 = 2*Rand_Uniform(0,1)-1
        w   = X_1*X_1+X_2*X_2
    
    Z_1 = pow_double(-2*log(w)/w,1/2)*X_1

    Z_2 = pow_double(-2*log(w)/w,1/2)*X_2
    
    Array[0] = mean + sigma*Z_1
    
    Array[1] = mean + sigma*Z_2
    
' Generate N normal distributed random numbers '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef double* Uniform_Random_num(int N,double mean, double sigma)nogil:
    
    cdef double* Random_num_Array =<double*>malloc(sizeof(double)*N)
    
    cdef int k 
     
    for k in range(N):
    
        Random_num_Array[k] = Rand_Uniform(mean,sigma)
    
    return &Random_num_Array[0]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef double* Normal_Random_num(int N,double mean, double sigma) nogil:

    cdef double* Random_num_Array =<double*>malloc(sizeof(double)*N)
    
    cdef int k 
    
    for k in range(N):
        
        Random_num_Array[k] = 0

    for k in range(N//2):
        
        Assign_Box_Muller(&Random_num_Array[k*2],mean,sigma)
    
    if N%2 == 1:
        
        Random_num_Array[N-1] = Box_Muller_Normal_Dis(mean,sigma)

    return &Random_num_Array[0]

' All Input Vectors v must satisfy ||v|| = 1  '
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
    
cdef double* Rotation_Euler(double* Input_vec,double* Vec,double angle) nogil:
    
    cdef int k
    
    cdef double* Cross_prod  = <double*>malloc(sizeof(double)*3)
    
    cdef double* Cross_prod_2  = <double*>malloc(sizeof(double)*3)
    
    cdef double* Ref_Axis    = <double*>malloc(sizeof(double)*3)
    
    cdef double* Cross_Vec   = <double*>malloc(sizeof(double)*3)
    
    Ref_Axis[0] = 0
    
    Ref_Axis[1] = 0
    
    Ref_Axis[2] = 1
    
    cdef double Rot_Angle = 0
    
    cdef double Dot_prod         = 0
    
    cdef double Norm             = 0
    
    cdef int dim = 3
    
    cdef int stride = 1
    
    ' Cross Prod_1 '
    
    Cross_prod[0] = Input_vec[1]*Ref_Axis[2]-Ref_Axis[1]*Input_vec[2]
        
    Cross_prod[1] = Input_vec[2]*Ref_Axis[0]-Ref_Axis[2]*Input_vec[0]
    
    Cross_prod[2] = Input_vec[0]*Ref_Axis[1]-Ref_Axis[0]*Input_vec[1]
    
    Norm = pow_double(ddot(&dim,&Cross_prod[0],&stride,&Cross_prod[0],&stride),1/2)
    
    for k in range(3):
        
        Cross_prod[k] = Cross_prod[k]/Norm
        
    ' Cross Prod_2 '
    
    
    Cross_Vec[0] = Cross_prod[1]*Vec[2]-Vec[1]*Cross_prod[2]
        
    Cross_Vec[1] = Cross_prod[2]*Vec[0]-Vec[2]*Cross_prod[0]
    
    Cross_Vec[2] = Cross_prod[0]*Vec[1]-Vec[0]*Cross_prod[1]
    
    Cross_prod_2[0] = Cross_Vec[1]*Cross_prod[2]-Cross_prod[1]*Cross_Vec[2]
        
    Cross_prod_2[1] = Cross_Vec[2]*Cross_prod[0]-Cross_prod[2]*Cross_Vec[0]
    
    Cross_prod_2[2] = Cross_Vec[0]*Cross_prod[1]-Cross_prod[0]*Cross_Vec[1]
    
    Dot_prod = ddot(&dim,&Cross_prod[0],&stride,&Vec[0],&stride)
    
    Rot_Angle     = acos(ddot(&dim,&Input_vec[0],&stride,&Ref_Axis[0],&stride)) 
    
    cdef double* Vec_Rotated = <double*>malloc(sizeof(double)*3)
    
    for k in range(3):
        
        Vec_Rotated[k] = Cross_prod[k]*Dot_prod+cos(Rot_Angle)*Cross_prod_2[k]+sin(Rot_Angle)*Cross_Vec[k]
    
    free(Input_vec)
    free(Cross_prod)
    free(Ref_Axis)
    free(Cross_Vec)
    free(Cross_prod_2)
    
    return &Vec_Rotated[0]

' Get Rotation Matrix from Rotation vector and angle '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
    
cdef double* Rot_Mat_from_Rot_Axis(double* Rot_vec,double angle) nogil:
    
    cdef int k
    
    cdef double* Rot_mat  = <double*>malloc(sizeof(double)*9)
    
    cdef double* Vec      = <double*>malloc(sizeof(double)*3)
    
    ' Normalize Vector '
    
    cdef double Norm = 0
    
    for k in range(3):
        
        Norm = Norm + Rot_vec[k]*Rot_vec[k]
        
    for k in range(3):
        
        Vec[k] = Rot_vec[k]/(pow_double(Norm,1/2))
        
    ' Define Rotation Matrix '
    
    Rot_mat[0] = cos(angle)+(Vec[0]*Vec[0])*(1-cos(angle))
    Rot_mat[1] = Vec[0]*Vec[1]*(1-cos(angle))-Vec[2]*sin(angle)
    Rot_mat[2] = Vec[2]*Vec[0]*(1-cos(angle))+Vec[1]*sin(angle)
    Rot_mat[3] = Vec[1]*Vec[0]*(1-cos(angle))+Vec[2]*sin(angle)
    Rot_mat[4] = cos(angle)+Vec[1]*Vec[1]*(1-cos(angle))
    Rot_mat[5] = Vec[1]*Vec[2]*(1-cos(angle))-Vec[0]*sin(angle)
    Rot_mat[6] = Vec[2]*Vec[0]*(1-cos(angle))-Vec[1]*sin(angle)
    Rot_mat[7] = Vec[2]*Vec[1]*(1-cos(angle))+Vec[0]*sin(angle)
    Rot_mat[8] = cos(angle)+Vec[2]*Vec[2]*(1-cos(angle))
    
    free(Vec)
                  
    return &Rot_mat[0]
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef double* Get_Rot_mat(double angle, int axis) nogil:
    
    cdef int k,l
    
    cdef double* Return_Rot_mat = <double*>malloc(sizeof(double)*9)
    
    for k in range(3):
        
        for l in range(3):
            
            Return_Rot_mat[k*3+l] = 0
            
    if axis == 0:
        
        Return_Rot_mat[0]     = 1
        
        Return_Rot_mat[3+1]   = cos(angle)
        
        Return_Rot_mat[3+2]   = -sin(angle)
        
        Return_Rot_mat[3*2+1] = sin(angle)
        
        Return_Rot_mat[3*2+2] = cos(angle)
        
    elif axis == 1:
        
        Return_Rot_mat[3+1]   = 1
        
        Return_Rot_mat[0]     = cos(angle)
        
        Return_Rot_mat[2]     = sin(angle)
        
        Return_Rot_mat[3*2]   = -sin(angle)
        
        Return_Rot_mat[3*2+2] = cos(angle)
        
    elif axis == 2:
        
        Return_Rot_mat[3*2+2] = 1
        
        Return_Rot_mat[0]     = cos(angle)
        
        Return_Rot_mat[1]     = -sin(angle)
        
        Return_Rot_mat[3]     = sin(angle)
        
        Return_Rot_mat[3+1]   = cos(angle)
        
    return &Return_Rot_mat[0]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef void Random_walk_SE_3(double [:] Init_Or, double [:] Init_pos, double D_33,double D_44,\
                           int N, double time,double* Random_walk_pos,double* Random_walk_or,str Stoch_Pr = 'Contour_Enh')nogil:
    
    cdef double* Random_or = <double*>malloc(sizeof(double)*3)
    
    cdef double* Cross_prod  = <double*>malloc(sizeof(double)*3) 
    
    cdef double* Euler_y_mat = <double*>malloc(sizeof(double)*3*3)
    
    cdef double* Euler_z_mat = <double*>malloc(sizeof(double)*3*3)
                  
    cdef double* Rot_mat     = <double*>malloc(sizeof(double)*3*3)
    
    cdef double* Mat_Prod  = <double*>malloc(sizeof(double)*3*3)              
    
    cdef double* Mat_Prod_Out  = <double*>malloc(sizeof(double)*3*3) 
    
    cdef int k, l
    
    for k in range(3):
    
        Random_walk_pos[k] = 0
    
    for k in range(9):
                  
        Mat_Prod[k] = 0
        
        Rot_mat[k]  = 0
        
        Euler_y_mat[k] = 0
        
        Euler_z_mat[k] = 0
        
        Mat_Prod_Out[k] = 0
        
    for k in range(3):
        
        Random_or[k]       = Init_Or[k]
        
        Random_walk_pos[k] = Init_pos[k]
    
    cdef double beta_pr = 0
      
    cdef int stride = 1
    
    cdef int dim    = 3
    
    cdef double alpha = 1
    
    cdef double Rand_sp = 0
                  
    cdef double theta  
    
    cdef double Norm            
    
    ' Initilize random normal distributions '
    
    cdef double* Array_eps 
    
    if Stoch_Pr == 'Contour_Enh':

        Array_eps = Normal_Random_num(N,0,1) 

    elif Stoch_Pr == 'Contour_Compl':
        
        Array_eps = <double*>malloc(sizeof(double)*N)

        for k in range(N):

            Array_eps[k] = 1.0/sqrt(2*D_33)

    cdef double* Array_beta 
    
    Array_beta = Normal_Random_num(N,0,1)
        
    cdef double* Array_gamma
    
    Array_gamma = Uniform_Random_num(N,-pi,2*pi)
    
    cdef double* Mat_Vec_Store = <double*>malloc(sizeof(double)*3) 
    
    for k in range(N):
        
        ' Update Random Orientation '
                  
        Cross_prod[0] = -1*Random_or[1]
        Cross_prod[1] = Random_or[0]*1
        Cross_prod[2] = 0
        
        Rand_sp = sqrt(2*time*D_33/N)*Array_eps[k]
        
        ' Update Random Position '
                  
        daxpy(&dim,&Rand_sp,&Random_or[0],&stride,&Random_walk_pos[0],&stride)
        
        Rand_sp = sqrt(2*time*D_44/N)*Array_beta[k]
        
        free(Euler_y_mat)
        
        Euler_y_mat = Get_Rot_mat(Rand_sp,1)
        
        free(Euler_z_mat)
        
        Euler_z_mat = Get_Rot_mat(Array_gamma[k],2)
        
        theta = pow_double(Cross_prod[0]*Cross_prod[0]+Cross_prod[1]*Cross_prod[1]\
                           +Cross_prod[2]*Cross_prod[2],0.5)
        
        theta = asin(theta)
        
        free(Rot_mat)
        
        Rot_mat = Rot_Mat_from_Rot_Axis(&Cross_prod[0],theta)
        
        dgemm('N','N',&dim,&dim,&dim,&alpha,&Rot_mat[0],&dim,&Euler_z_mat[0],&dim,&beta_pr,&Mat_Prod[0],&dim)
        
        dgemm('N','N',&dim,&dim,&dim,&alpha,&Mat_Prod[0],&dim,&Euler_y_mat[0],&dim,&beta_pr\
                  ,&Mat_Prod_Out[0],&dim)
        
        dgemm('N','T',&dim,&dim,&dim,&alpha,&Mat_Prod[0],&dim,&Rot_mat[0],&dim,&beta_pr,&Mat_Prod_Out[0],&dim)
        
        dgemv('N',&dim,&dim,&alpha,&Mat_Prod_Out[0],&dim,&Random_or[0],&stride,&beta_pr,&Mat_Vec_Store[0],&stride)
        
        for l in range(3):
            
            Random_or[l]          = Mat_Vec_Store[l]
            
            Random_walk_or[l] = Random_or[l] 
            
    free(Euler_z_mat)
    free(Euler_y_mat)
    free(Rot_mat)            
    free(Mat_Prod)          
    free(Cross_prod)
    free(Random_or)
    free(Array_beta)
    free(Array_gamma)
    free(Array_eps)
    free(Mat_Vec_Store)
    free(Mat_Prod_Out)

'  Sort Index List without output '   
    
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

cdef extern from "math.h":
    double acos(double) nogil
    
cdef struct IndexedElement:
    cnp.ulong_t index
    cnp.float64_t value

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
    
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
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef (int) Get_Dir_Index(double* Input,double [:,:] Dir_Array) nogil:

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
        
        ptr_or      = &Dir_Array[k,:][0]
        
        Distance[k] = acos(ddot(dim,&Input[0],stride_or,ptr_or,stride_or))
        
    argsort_carray(Distance,&index_order[0],Length)
    
    Return_ind = index_order[0]
    
    free(Distance)
    free(index_order)
    free(stride_or)
    free(dim)
    
    return Return_ind

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef Random_walk_Kernel_Cython(double [:] Init_pos, double [:] Init_Or, int Rand_walk_iter,double D_33,double D_44,\
                    int Walk_Length, double time,int Sp_size, int Anglular_Size,double [:,:] Or_Array,str Method):
    
    assert Method == 'Contour_Enh' or Method == 'Contour_Compl', ' Input Method not available'
    
    cdef double Kernel_size =  30.0/(pow_double(D_33*time*12.0/Walk_Length,0.5))

    cdef double Bin_Area_Sphere = 2*pi/Anglular_Size
    
    cdef int* End_Points_Walk = <int*>malloc(sizeof(int)*Rand_walk_iter*4)
    
    cdef double [:,:,:,:] Kernel = np.zeros((Sp_size,Sp_size,Sp_size,Anglular_Size))
    
    cdef int k,l,m,n,x,y,z
    
    cdef double* Walk  = <double*>malloc(sizeof(double)*Rand_walk_iter*3)
    
    cdef double* Walk_or  = <double*>malloc(sizeof(double)*Rand_walk_iter*3)

    for l in prange(Rand_walk_iter,nogil = True):
        
        Random_walk_SE_3(Init_Or, Init_pos,D_33, D_44,\
                         Walk_Length, time,&Walk[l*3],&Walk_or[l*3],Method)
        
        '  Move the random walk start to the volume center and save Endpoint '
        
        End_Points_Walk[l*4] = Sp_size//2+round(Walk[l*3]*Kernel_size)
      
        End_Points_Walk[l*4+1] = Sp_size//2+ round(Walk[l*3+1]*Kernel_size)
        
        End_Points_Walk[l*4+2] = Sp_size//2+round(Walk[l*3+2]*Kernel_size)
        
        End_Points_Walk[l*4+3] = Get_Dir_Index(&Walk_or[l*3],Or_Array)
        ' Iterate over Kernel '
                            
        Kernel[End_Points_Walk[l*4],End_Points_Walk[l*4+1],End_Points_Walk[l*4+2],End_Points_Walk[l*4+3]] += 1.0/(Bin_Area_Sphere*Rand_walk_iter*Kernel_size*Kernel_size\
                                                 *Kernel_size)
                                                            
    free(End_Points_Walk)
    free(Walk_or)
    free(Walk)

    
    return np.array(Kernel)/(np.sum(np.array(Kernel)))