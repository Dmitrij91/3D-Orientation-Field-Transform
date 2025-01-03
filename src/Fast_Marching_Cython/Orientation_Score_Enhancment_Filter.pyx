import numpy as np
import cython
cimport numpy as cnp
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from openmp cimport omp_get_thread_num
from libc.math cimport signbit,exp, pi,fmin,sqrt
from cython.view cimport array as cvarray
from libc.string cimport memcpy
from scipy.linalg.cython_lapack cimport dgeqrf
from scipy.linalg.cython_blas cimport dger
from scipy.linalg.cython_blas cimport ddot

cdef extern from "math.h" nogil:
    double fmax(double,double)
    double cos(double)
    double sin(double)
    double atan2(double x,double y) nogil
    double acos(double x) nogil
    double pow(double,int)
    double abs(double) nogil
    int round(double)
    
cdef extern from "math.h"nogil:
    
    cdef double pow_double "pow"(double,double)
    cdef int fmin_int "fmin"(int,int)
    cdef int fmax_int "fmax"(int,int)


' Vessel Intensity Meausre '

###############################################################
## Input: Discrete Coordinates, Angles , Orienation Vector   ##
##        Radius, theta                                      ##
## Output: Product of Vesselintensity in direction theta     ##
##         and and 180+theta at radius r                     ##
###############################################################
#cpdef Vessel_Intensity(double [:] data,cnp.ndarray[int,ndim = 1] data_dir, cnp.ndarray[int,ndim = 1,\
#                                        negative_indices = False] indices,cnp.ndarray[int,ndim = 1,\
#                                        negative_indices = False] indptr, cnp.ndarray[double,ndim = 1]\
#                                      direction,double angle, double radius,int Center_Vox):
    
' Return Orthogonal unit vertor in direction alpha within the orthogonal plane to input direction  ' 
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


cdef double* Orth_Plane(double[:] direction,double alpha) nogil:
    
    
    ' Intitialization '
    
    cdef double Intensity = 0
    
    cdef int i,j 

    cdef double* Q = <double*>malloc(sizeof(double)*3*3)

    for i in range(3*3):

        Q[i] = 0

    for j in range(3):

        Q[j*3+j] = 1


    cdef double* outer        = <double*>malloc(sizeof(double)*3)
    
    cdef double* orth_vec_new = <double*>malloc(sizeof(double)*3)
    
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


    return &orth_vec_new[0]


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

    
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)  

cdef (double,double) Orientation_to_Euler(double* orientation) nogil:

    cdef double beta  = 0
    
    cdef double gamma = 0
    
    #print(f'--Input_2"{orientation[0]}""--')
    
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

cdef (double*,int) Get_Discrete_Dir(double* Input,double [:,:] Dir_Array) nogil:

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
    
    cdef double* Input_new = <double*>malloc(sizeof(double)*3)
    
    cdef int* index_order =<int*>malloc(sizeof(int)*Length) 
    
    for k in range(Length):
        
        ptr_or      = &Dir_Array[k,:][0]
        
        Distance[k] = acos(ddot(dim,&Input[0],stride_or,ptr_or,stride_or))
        
    
    argsort_carray(Distance,&index_order[0],Length)
    
    for k in range(3):
        
        Input_new[k] = Dir_Array[index_order[0]][k]
    
    Return_ind = index_order[0]
    
    free(Distance)
    free(index_order)
    free(stride_or)
    free(dim)
    
    return &Input_new[0], Return_ind


' Get nearest spatial positition '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef int* Get_Spatial_Pos(double* Ref_Coord, double Radius,double* Direction) nogil:
    
    cdef int* Vec_tr = <int*>malloc(sizeof(int)*3)
    
    cdef int k 
    
    for k in range(3):
        
        Vec_tr[k] = round(Ref_Coord[k]+Radius*Direction[k])
    
    return &Vec_tr[0]



' Compute Enhanced Intensity '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
    
cdef double Enh_Intensity(double Or_Data,double [:,:,:,:] Orientation_Score,double* Node_Coord,\
                         double [:,:] Dir_Array,double [:] direction, double* Radius,\
                         double* alpha, int boundary_pos) nogil:
    
    ' Initialization '        
    
    cdef double* Vec_Tr_pt
    
    cdef double Data = 0
    
    cdef double* Orth_Vec   
    
    cdef double* Orth_Vec_dis   
    
    cdef double* Orth_Vec_pt
    
    cdef int* Vec_Tr_pos        
    
    cdef int* Vec_Tr_pos_pt
    
    cdef int* Vec_Tr_neg        
    
    cdef int* Vec_Tr_neg_pt
    
    cdef int index
    
    cdef int index_neg
    
    cdef int k,l

    Orth_Vec    = Orth_Plane(direction,alpha[0])
    
    cdef double* Orth_Vec_neg = <double*>malloc(sizeof(double)*3)

    for k in range(3):

        Orth_Vec_neg[k] = -Orth_Vec[k]

    Orth_Vec_pt = &Orth_Vec[0]

    ' Turn Continuous orthogonal vector into discrete Vector on Grid '
    
    Orth_Vec_dis, index   = Get_Discrete_Dir(Orth_Vec_pt,Dir_Array) 

    ' Get Coordinates of the translated vector of input Radius '
    
    #print(f'--"{Node_Coord[0]}""--')
    
    Vec_Tr_pos        = Get_Spatial_Pos(Node_Coord, Radius[0],&Orth_Vec[0])
    
    Vec_Tr_neg        = Get_Spatial_Pos(Node_Coord, Radius[0],&Orth_Vec_neg[0])
    
    cdef int min_check, min_check_1
    
    cdef int max_check, max_check_1
    
    min_check = Vec_Tr_pos[0] 
    
    min_check_1 = Vec_Tr_neg[0]
    
    max_check = Vec_Tr_pos[0] 
    
    max_check_1 = Vec_Tr_neg[0]
    
    for k in range(1,3):
        
        if min_check > Vec_Tr_pos[k]:
            
            min_check = Vec_Tr_pos[k]
            
        if min_check_1 > Vec_Tr_neg[k]:
            
            min_check_1 = Vec_Tr_neg[k]
            
        if max_check < Vec_Tr_pos[k]:
            
            max_check = Vec_Tr_pos[k]
            
        if max_check_1 < Vec_Tr_neg[k]:
            
            max_check_1 = Vec_Tr_neg[k]
    
    
    if fmax_int(max_check,max_check_1) < boundary_pos and fmin_int(min_check,min_check_1) >= 0:
        
        Data = fmax(Orientation_Score[Vec_Tr_pos[0],Vec_Tr_pos[1],Vec_Tr_pos[2],index],0)\
                *fmax(Orientation_Score[Vec_Tr_neg[0],Vec_Tr_neg[1],Vec_Tr_neg[2],index],0)
            
    else:
 
        Data = Or_Data
    
    free(Vec_Tr_neg)
    free(Vec_Tr_pos)
    free(Orth_Vec_dis)
    free(Orth_Vec)
    free(Orth_Vec_neg)
    
    return Data

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
    
cdef double Angular_Reg_Kernel(double alpha,int trunc_num) nogil:
        
    cdef double output 
    
    cdef double sigma = pi/8
    
    cdef int k
        
    for k in range(trunc_num):
        
        output +=(1.0/(sqrt(2*pi)*sigma))*exp(-pow((alpha+2*k*pi),2)/(2*pow(sigma,2)))
            
    return output

' theta in [0,2pi], Array_Angle in [0,pi] '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef double Sum_Integral_min_Intensity(double [:] data_angle,\
                            int Angles_Num,int angle_precision, int Dscr_Int) nogil:
    
    cdef int k,l
    
    cdef double Int_Sum
    
    cdef double Int_Sum_min 
    
    ' Compute min Response over theta '
    
    cdef double* Theta_Array  = <double*>malloc(sizeof(double)*angle_precision)
    
    for k in range(angle_precision):
        
        for l in range(Dscr_Int):
        
            Int_Sum += (Angular_Reg_Kernel(k*(2*pi/angle_precision) - l*pi/Dscr_Int,1)*data_angle[l])/Dscr_Int
        
        Theta_Array[k] = Int_Sum
        
        Int_Sum = 0
    
    Int_Sum_min = Theta_Array[0]
    
    ' Minimize over angle[0,2*pi] '

    for k in range(1,angle_precision):
        
        if Int_Sum_min > Theta_Array[k]:
        
            Int_Sum_min = Theta_Array[k]
    
    free(Theta_Array)
    
    return Int_Sum_min

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef Enhancement_Filter_Orientation_Score(double [:,:,:,:] Orient_Score,double [:,:] Dir_Array,int Rad_num,\
                            int Num_Theta,double Rad_space):
    
    cdef int s,k,l,m,n,r,d,X,Y,Z,Num_angle
    
    X = Orient_Score.shape[0]
    
    Y = Orient_Score.shape[1]
    
    Z = Orient_Score.shape[2]
    
    Num_angle = Orient_Score.shape[3]
    
    cdef double [:,:,:,:,:,:] Data_Enh = np.zeros((X,Y,Z,Num_angle,Rad_num,Num_Theta))
    
    cdef double* Array_Theta = <double*>malloc(sizeof(double)*Num_Theta)
    
    cdef double* Rad_Array = <double*>malloc(sizeof(double)*Rad_num)
    
    cdef double* Radius_pt

    for k in range(Rad_num):
    
        Rad_Array[k] = 0.8+k*Rad_space
    
    cdef double* Array_Theta_pt
    
    for k in range(Num_Theta):
        
        Array_Theta[k] = k*(pi/(Num_Theta-1))   

    cdef double* Node_Coord = <double*>malloc(sizeof(double)*3*X*Y*Z)

    cdef int X_id,Y_id,Z_id

    for k in prange(X*Y*Z,schedule = 'static', nogil = True):
        
        X_id = k/(Y*Z)

        Y_id = (k%(Y*Z))/Z

        Z_id = (k%(Y*Z))%Z

        Node_Coord[3*X_id*Y*Z+3*Y_id*Z+3*Z_id] = X_id
                        
        Node_Coord[3*X_id*Y*Z+3*Y_id*Z+3*Z_id+1] = Y_id
                            
        Node_Coord[3*X_id*Y*Z+3*Y_id*Z+3*Z_id+2] = Z_id
                
                
        for m in range(Num_angle):

            for r in range(Rad_num):
                
                for d in range(Num_Theta):
                    
                    Node_Coord_pt  = &Node_Coord[3*X_id*Y*Z+3*Y_id*Z+3*Z_id]
                    
                    Array_Theta_pt = &Array_Theta[d]
                    
                    Radius_pt      = &Rad_Array[r] 
                    
                    Data_Enh[X_id,Y_id,Z_id,m,r,d] = Enh_Intensity(Data_Enh[X_id,Y_id,Z_id,m,r,d],Orient_Score,Node_Coord_pt,\
                                            Dir_Array,Dir_Array[m],Radius_pt,Array_Theta_pt,X)
    free(Node_Coord)
    free(Array_Theta)
    free(Rad_Array)
    
    return np.array(Data_Enh)

' Filter Response disk boundary '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef Filter_Enh_angle_sum(double [:,:,:,:,:,:] Enh_data, int angle_prec,int Dscr_Int):
    
    cdef int X,Y,Z,Dir_num,Rad_Num,Angle_dim 
    
    X = Enh_data.shape[0]
    
    Y = Enh_data.shape[1]
                       
    Z = Enh_data.shape[2]
    
    Dir_num = Enh_data.shape[3]
    
    Rad_Num = Enh_data.shape[4]
    
    Angle_dim = Enh_data.shape[5]
    
    cdef double[:,:,:,:,::1] Enhanced_Volume = np.zeros((X,Y,Z,Dir_num,Rad_Num))
    
    cdef int k,l,m,n,r,p
    
    for k in prange(X,nogil = True):
        
        for l in range(Y):
            
            for m in range(Z):
                
                for n in range(Dir_num):
                    
                    for r in range(Rad_Num):
                        
                        Enhanced_Volume[k,l,m,n,r] = Sum_Integral_min_Intensity(Enh_data[k,l,m,n,r,:],\
                                                    Angle_dim,angle_prec,Dscr_Int)
    
    return Enhanced_Volume