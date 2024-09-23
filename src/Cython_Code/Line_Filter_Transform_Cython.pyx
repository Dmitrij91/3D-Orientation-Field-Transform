import numpy as np
import cython
cimport numpy as cnp
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from openmp cimport omp_get_max_threads, omp_get_thread_num
from libc.math cimport signbit,exp, pi,pow, fmax,fmin,sqrt,abs
from cython.view cimport array as cvarray
from libc.string cimport memcpy
    
' Sorting indices and values with return variables '
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef Sort(cnp.ndarray[double,negative_indices = True] List,cnp.ndarray[int,negative_indices = False] Indices ):
    cdef int Length              = List.shape[0]
    cdef int S_Ind               = 0
    cdef cnp.ndarray[double,ndim = 1,negative_indices = False] Sorted_List = np.zeros(Length)
    cdef cnp.ndarray[int,ndim = 1,negative_indices = False] Sorted_Indices = np.zeros(Length,dtype = np.int32)
    
    'Help Variables'
    
    cdef int l,k,s
    cdef double S_var = 0
    
    'Copy Array'
    
    for s in prange(Length,nogil = True):
        Sorted_List[s]    = List[s]
        Sorted_Indices[s] = s
        
    for k in range(Length):
        S_var = Sorted_List[k]
        S_Ind = Sorted_Indices[k]
        
        for l in range(k+1,Length):
            if S_var > Sorted_List[l]:
                S_var = Sorted_List[l]
                S_Ind = Sorted_Indices[l]
                
                'Outmemory the smallest value'
                
                Sorted_List[l] = Sorted_List[k]
                Sorted_List[k] = S_var
                Sorted_Indices[l] = Sorted_Indices[k]
                Sorted_Indices[k] = S_Ind
    for s in prange(Length,nogil = True):
        Indices[s] = Indices[Sorted_Indices[s]]

    return Sorted_List,Indices
    
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

cpdef void argsort(cnp.float64_t[:] data, int [:] order):
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
 

' Overwrite distance into indecices for each graph orientation '
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)    

cdef inline double argsort_c_array(double* data,int* indices,int stride_dir\
                                   ,int or_index,int stride_or) nogil:
    
    ' Initilialization  '
    
    cdef int i,j

    # Allocate index tracking array.
    
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(stride_dir * sizeof(IndexedElement))
    
    
    ' Sum over orientations on graph for Or or_index '
            
    for i in range(stride_dir):
            
        order_struct[i].index = i
            
        order_struct[i].value = data[i*stride_or]
        
    # Sort index tracking array.
    
    qsort(<void *> order_struct, stride_dir, sizeof(IndexedElement), _compare)
    
    # Copy indices from index tracking array to output array.
    
    for j in range(stride_dir):
        #print(str(j)+'Test')
        indices[or_index*stride_dir+j] = order_struct[j].index
        
    # Free index tracking array.

    free(order_struct)


' Cython nogil --> replace numpy initialization by c arrays,\
function call nogil within prange loop memoryview possible '

' Function call for Line_Filter Transform '

##################################################
# Input: Integer valued indices of 3D grid       #
#       : Indices,indtptr of in sparse csr format#
#       : List if normalized directions          #
# Output: Sorted Array of indices giving         #
# indices of nearest directions to 'directions'  # 
##################################################

' Import Innerproduct from blas library '

from scipy.linalg.cython_blas cimport ddot
#cdef extern from 'cblas.h':
#    double ddot 'cblas_ddot'(int N,
#                             double* X, int incX,
#                             double* Y, int incY) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef inline double Get_Neighbour_dir_cython(int* data_dir,int* indices,int Length_patch,\
                                             double* direction,int Voxel, int* indices_pos, \
                                            int* indices_neg,int Or_size) nogil:

    ' Get Position of center patch '
    
    
    cdef int Center_Vox          = (Length_patch-1)/2
    
    #print((indptr[Voxel+1]-indptr[Voxel]-1)/2)
    
    cdef int Patch_Size          = Length_patch-1
    
    cdef int k,l,ori,dirc,j
    
    cdef double* ptr_or_dir
    
    cdef double* ptr_or
    
    cdef double* ptr_dir
    
    cdef double* ptr_or_dis
    
    cdef double* ptr_or_dis_neg
    
    cdef double* ptr_or_neg
    
    cdef int It  = 0
    
    cdef int* stride_dir  = < int*>malloc(sizeof(int))
    
    cdef int* stride_or   = < int*>malloc(sizeof(int)) 
    
    cdef int* dim         = < int*>malloc(sizeof(int))
    
    ' Assign strides for inner product '
    
    dim[0]        = 3
    
    stride_dir[0] = 1
    
    stride_or[0]  = 1
    
    ' The Directions must lie linear in memory '
    
    cdef double* directions = < double*>malloc(sizeof(double)*(Patch_Size)*3)
    
    ' Initialize Distance '
    
    cdef double* distance     = < double*>malloc(sizeof(double)*(Patch_Size+1)*Or_size)
    cdef double* distance_neg = < double*>malloc(sizeof(double)*(Patch_Size+1)*Or_size)
    
    ' Copy Orientation into linear memory '
    
    cdef double* orientations     = < double*>malloc(sizeof(double)*Or_size*3)
    
    cdef double* orientations_neg = < double*>malloc(sizeof(double)*Or_size*3)
    
    for ori in range(Or_size):
        
        orientations[ori*3]   = direction[ori*3]
        
        orientations[ori*3+1] = direction[ori*3+1]
        
        orientations[ori*3+2] = direction[ori*3+2]
        
        orientations_neg[ori*3]   = -direction[ori*3]
        
        orientations_neg[ori*3+1] = -direction[ori*3+1]
        
        orientations_neg[ori*3+2] = -direction[ori*3+2]
    
    for k in range(Patch_Size+1):
        
        ' Remove center patch '
        
        if k != Center_Vox:
            
        
            directions[It*3]   = (data_dir[Center_Vox*3] - \
                                         data_dir[k*3])/sqrt(pow(
                                         data_dir[Center_Vox*3] - \
                                         data_dir[k*3],2)+\
                                         pow(data_dir[Center_Vox*3+1] - \
                                         data_dir[k*3+1],2)+pow(\
                                         data_dir[Center_Vox*3+2] - \
                                         data_dir[k*3+2],2))
    
    
    
            directions[It*3+1] = (data_dir[Center_Vox*3+1]- \
                                         data_dir[k*3+1])/sqrt(pow(
                                         data_dir[Center_Vox*3] - \
                                         data_dir[k*3],2)+\
                                         pow(data_dir[Center_Vox*3+1] - \
                                         data_dir[k*3+1],2)+pow(\
                                         data_dir[Center_Vox*3+2] - \
                                         data_dir[k*3+2],2))

            directions[It*3+2] = (data_dir[Center_Vox*3+2]- \
                                         data_dir[k*3+2])/sqrt(pow(
                                         data_dir[Center_Vox*3] - \
                                         data_dir[k*3],2)+\
                                         pow(data_dir[Center_Vox*3+1] - \
                                         data_dir[k*3+1],2)+pow(\
                                         data_dir[Center_Vox*3+2] - \
                                         data_dir[k*3+2],2))
            
            
            ptr_dir     = &directions[It*3]
            
            
            for l in range(Or_size):
                
                ' Iterate over orientation list '
                
                ptr_or      = &orientations[l*3]
                            
                ptr_or_neg  = &orientations_neg[l*3]
                
                distance[It*Or_size+l]  = acos(ddot(dim,ptr_dir,stride_dir,ptr_or,stride_or))
                
                distance_neg[It*Or_size+l]  = acos(ddot(dim,ptr_dir,stride_dir,ptr_or_neg,stride_or))
            
            It = It + 1
            
        elif k == Center_Vox: 
            
            ' Move to Center_Vox to last pos '
            
            for j in range(Or_size):
                
                distance[(Patch_Size)*Or_size+j]  = 100
                
                distance_neg[(Patch_Size)*Or_size+j]  = 100
    
    ' Allocate minimal indices '
            

# #        #print(distance[k])      
    
    for dirc in range(Or_size):
    
        ptr_or_dis = &distance[dirc]
        
        ptr_or_dis_neg = &distance_neg[dirc]
        
# #         #print(distance[dirc])
        
        argsort_c_array(ptr_or_dis, indices_pos,Length_patch,dirc,Or_size)
        
        argsort_c_array(ptr_or_dis_neg, indices_neg,Length_patch,dirc,Or_size)
        
#        # print(np.asarray(indices_pos)[0,0:5])
    
    ' Free allocated memory '
    
    free(directions)
    free(distance)
    free(distance_neg)
    free(orientations)
    free(orientations_neg)
    free(stride_dir)
    free(stride_or)
    free(dim)
    


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef void Max_Dir_Response_Cython(int* Index_pos,int* Index_neg,int Step_Size,\
                              double* neigh_patch, int Or_size, double [:] Filter_transform,\
                                   int Patch_Size) nogil:
    
    ' Initialization '
    
    cdef int Center_Vox           = (Patch_Size-1)/2
    
    cdef double max_value         = 0
    
    cdef double mean_value        = 0
    
    cdef double std_value         = 0
    
    cdef int max_index
    
    cdef int k,l,i
    
    cdef double* Help_Max_var = <double*> malloc(sizeof(double)*Or_size)
    
    cdef double* Help_Max_var_pt
    
    ' Add prange '
    
    for k in range(Or_size):
        
        Help_Max_var_pt = &Help_Max_var[0]
        
        Help_Max_var_pt[k] = neigh_patch[Center_Vox]
        
        for l in range(Step_Size):
            
            Help_Max_var_pt[k] = Help_Max_var_pt[k] + neigh_patch[Index_pos[Patch_Size*k+l]]+neigh_patch[Index_neg[Patch_Size*k+l]]
        
        ' Compute Mean_Value over Orientations '
        
        mean_value += Help_Max_var_pt[k]/Or_size
        
    ' Compute Standart deviation '    
    
    for k in range(Or_size):
        
        std_value += (mean_value -Help_Max_var_pt[k])*(mean_value -Help_Max_var_pt[k])/Or_size
         
    ' Return maximal value, index '
    
    for i in range(Or_size):       
        
        Help_Max_var_pt = &Help_Max_var[i]
        
        if Help_Max_var_pt[0] > max_value:
            
            max_value = Help_Max_var_pt[0]
            
            max_index = i          
            
    ' Copy result into LF_transform memoryview '
    
    Filter_transform[0] = max_value
    
    Filter_transform[2] = mean_value
    
    Filter_transform[3] = sqrt(std_value)
    
    Filter_transform[1] = max_index
    
    free(Help_Max_var)
            
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef Main_Line_Filter_Transform_Cython(double [:] data,cnp.ndarray[int,ndim = 1] \
                                      data_dir, cnp.ndarray[int,ndim = 1,\
                                        negative_indices = False] indices,cnp.ndarray[int,ndim = 1,\
                                        negative_indices = False] indptr, cnp.ndarray[double,ndim = 1]\
                                      orientations, cnp.ndarray[int,ndim = 1,negative_indices = False] window):

    ' Initialization '
    
    cdef int voxel,l,k,i,j
    
    ' Define Enhancement Size '
    
    cdef int Step_Size = (window[0]-1)/2
    
    cdef int Length_patch 
    
    cdef int Or_size       = int(orientations.shape[0]/3) 
    
    cdef int Length    = indptr.shape[0]-1
    
    ' Initialize Output: Returns Max_Value and index of Max direction '
    
    cdef double [:,::1] LF_Transform    = np.zeros((Length,4))
    
    
    cdef double* patch 
    
    cdef int* data_dir_pt
    
    cdef int* indices_pt
    
    cdef double* orientations_pt 
    
    cdef int* data_dir_pt_loop
    
    cdef int* indices_pt_loop
    
    cdef double* orientations_pt_loop
    
    cdef double* patch_loop
    
    cdef int* Ind_Pos
    
    cdef int* Ind_Neg 
    
    cdef int* Ptr_ind_pos
    
    cdef int* Ptr_ind_neg
    
    for voxel in prange(Length,nogil = True):
        
        Length_patch           = indptr[voxel+1]-indptr[voxel]
        
        patch = <double*> malloc(sizeof(double)*Length_patch)
        
        ' Copy patch data '
        
        for k in range(Length_patch):
            
             patch[k] = data[indices[indptr[voxel]+k]]
                
        patch_loop = &patch[0]
                
        data_dir_pt = <int*> malloc(sizeof(int)*Length_patch*3)
        
        ' Copy patch data '
        
        for i in range(Length_patch):
            
            data_dir_pt[i*3] = data_dir[3*indices[indptr[voxel]+i]]
                
            data_dir_pt[i*3+1] = data_dir[3*indices[indptr[voxel]+i]+1]
            
            data_dir_pt[i*3+2] = data_dir[3*indices[indptr[voxel]+i]+2]
            
        data_dir_pt_loop = &data_dir_pt[0]
        
        indices_pt = <int*> malloc(sizeof(int)*Length_patch)
        
        ' Copy patch data '
        
        for j in range(Length_patch):
            
             indices_pt[j] = indices[indptr[voxel]+j]
        
        ' Copy orientations data '
        
        indices_pt_loop = &indices_pt[0]
        
        orientations_pt = <double*> malloc(sizeof(double)*Or_size*3)
        
        for l in range(Or_size):
            
            orientations_pt[l*3] = orientations[l*3]
                
            orientations_pt[l*3+1] = orientations[l*3+1]
                
            orientations_pt[l*3+2] = orientations[l*3+2]
        
        orientations_pt_loop = &orientations_pt[0]
        
        Ind_Pos = <int*> malloc(sizeof(int)*Or_size*Length_patch)
        
        Ind_Neg = <int*> malloc(sizeof(int)*Or_size*Length_patch)
        
        Ptr_ind_pos = &Ind_Pos[0]
    
        Ptr_ind_neg = &Ind_Neg[0]
        
        Get_Neighbour_dir_cython(data_dir_pt_loop,indices_pt_loop,Length_patch,orientations_pt_loop,voxel,\
                                 Ptr_ind_pos,Ptr_ind_neg,Or_size)
        
        Max_Dir_Response_Cython(Ptr_ind_pos,Ptr_ind_neg,Step_Size,\
                              patch_loop, Or_size,LF_Transform[voxel],Length_patch)
        
        
        free(Ind_Pos)
        free(Ind_Neg)
        free(patch)
        free(orientations_pt)
        free(data_dir_pt)
        free(indices_pt)
        
    return np.asarray(LF_Transform)


' Orientation Alignment Filter '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef Orientation_Filter_Transform_Cython(double [:] LFT_Tr,int [:] OFT_Ind, cnp.ndarray[int,ndim = 1] \
                                      data_dir, cnp.ndarray[int,ndim = 1,\
                                        negative_indices = False] indices,cnp.ndarray[int,ndim = 1,\
                                        negative_indices = False] indptr, cnp.ndarray[double,ndim = 1]\
                                      orientations, cnp.ndarray[int,ndim = 1,negative_indices = False] window):

    ' Initialization '
    
    cdef int voxel,l,k,i,j
    
    ' Define Enhancement Size '
    
    cdef int Step_Size = (window[0]-1)/2
    
    cdef int Length_patch 
    
    cdef int Or_size       = int(orientations.shape[0]/3) 
    
    cdef int Length    = indptr.shape[0]-1
    
    ' Initialize Output: Returns Max_Value and index of Max direction '
    
    cdef double [:,::1] OF_Transform    = np.zeros((Length,4))
    
    
    cdef double* patch 
    
    cdef int* data_dir_pt
    
    cdef int* indices_pt
    
    cdef double* orientations_pt 
    
    cdef int* data_dir_pt_loop
    
    cdef int* indices_pt_loop
    
    cdef double* orientations_pt_loop
    
    cdef double* patch_loop
    
    cdef int* Ind_Pos
    
    cdef int* Ind_Neg 
    
    cdef int* Ptr_ind_pos
    
    cdef int* Ptr_ind_neg
    
    cdef int* LFT_Dir_ind_pt
    
    for voxel in prange(Length,nogil = True):
        
        Length_patch           = indptr[voxel+1]-indptr[voxel]
        
        patch = <double*> malloc(sizeof(double)*Length_patch)
        
        ' Copy patch data '
        
        for k in range(Length_patch):
            
             patch[k] = LFT_Tr[indices[indptr[voxel]+k]]
                
        patch_loop = &patch[0]
                
        data_dir_pt = <int*> malloc(sizeof(int)*Length_patch*3)
        
        ' Copy patch data '
        
        for i in range(Length_patch):
            
            data_dir_pt[i*3] = data_dir[3*indices[indptr[voxel]+i]]
                
            data_dir_pt[i*3+1] = data_dir[3*indices[indptr[voxel]+i]+1]
            
            data_dir_pt[i*3+2] = data_dir[3*indices[indptr[voxel]+i]+2]
            
        data_dir_pt_loop = &data_dir_pt[0]
        
        indices_pt = <int*> malloc(sizeof(int)*Length_patch)
        
        ' Copy patch data '
        
        for j in range(Length_patch):
            
             indices_pt[j] = indices[indptr[voxel]+j]
        
        ' Copy orientations data '
        
        indices_pt_loop = &indices_pt[0]
        
        orientations_pt = <double*> malloc(sizeof(double)*Or_size*3)
        
        for l in range(Or_size):
            
            orientations_pt[l*3] = orientations[l*3]
                
            orientations_pt[l*3+1] = orientations[l*3+1]
                
            orientations_pt[l*3+2] = orientations[l*3+2]
        
        orientations_pt_loop = &orientations_pt[0]
        
        Ind_Pos = <int*> malloc(sizeof(int)*Or_size*Length_patch)
        
        Ind_Neg = <int*> malloc(sizeof(int)*Or_size*Length_patch)
        
        Ptr_ind_pos = &Ind_Pos[0]
    
        Ptr_ind_neg = &Ind_Neg[0]
        
        Get_Neighbour_dir_cython(data_dir_pt_loop,indices_pt_loop,Length_patch,orientations_pt_loop,voxel,\
                                 Ptr_ind_pos,Ptr_ind_neg,Or_size)
        
        'Copy Direction indices '
        
        LFT_Dir_ind = <int*> malloc(sizeof(int)*Length_patch)
        
        for k in range(Length_patch):
            
            LFT_Dir_ind[k] = OFT_Ind[indices[indptr[voxel]+k]]
            
            #print(LFT_Dir_ind[k])
            
        LFT_Dir_ind_pt     = &LFT_Dir_ind[0]
        
        Max_Dir_OFT_Response_Cython(Ptr_ind_pos,Ptr_ind_neg,Step_Size,orientations_pt_loop,\
                              patch_loop,LFT_Dir_ind_pt, Or_size,OF_Transform[voxel],Length_patch)
        
        
        free(Ind_Pos)
        free(Ind_Neg)
        free(patch)
        free(orientations_pt)
        free(data_dir_pt)
        free(indices_pt)
        free(LFT_Dir_ind)
        
    return np.asarray(OF_Transform)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef void Max_Dir_OFT_Response_Cython(int* Index_pos,int* Index_neg,int Step_Size,double* orientations,\
                              double* neigh_patch,int* neigh_patch_OFT_dir, int Or_size, double [:] Filter_transform,\
                                   int Patch_Size) nogil:
    
    ' Initialization '
    
    cdef int Center_Vox           = (Patch_Size-1)/2
    
    cdef double max_value         = 0
    
    cdef double mean_value        = 0
    
    cdef double std_value         = 0
    
    cdef int max_index
    
    cdef int k,l,i
    
    cdef double* Help_Max_var = <double*> malloc(sizeof(double)*Or_size)
    
    cdef double* Help_Max_var_pt
    
    ' Add prange '
    
    for k in range(Or_size):
        
        Help_Max_var_pt = &Help_Max_var[0]
        
        Help_Max_var_pt[k] = neigh_patch[Center_Vox]*(2*pow((orientations[neigh_patch_OFT_dir[Center_Vox]*3]*\
                               orientations[k*3]+ orientations[neigh_patch_OFT_dir[Center_Vox]*3+1]*\
                               orientations[k*3+1]+orientations[neigh_patch_OFT_dir[Center_Vox]*3+2]*\
                               orientations[k*3+2]),2)-1)
        #print(Help_Max_var_pt[k])                  
        
        for l in range(Step_Size):
            
            Help_Max_var_pt[k] = Help_Max_var_pt[k] + neigh_patch[Index_pos[Patch_Size*k+l]]\
                                *(2*pow((orientations[neigh_patch_OFT_dir[Index_pos[Patch_Size*k+l]]*3]*\
                               orientations[k*3]+ orientations[neigh_patch_OFT_dir[Index_pos[Patch_Size*k+l]]*3+1]*\
                               orientations[k*3+1]+orientations[neigh_patch_OFT_dir[Index_pos[Patch_Size*k+l]]*3+2]*\
                               orientations[k*3+2]),2)-1)+neigh_patch[Index_neg[Patch_Size*k+l]]*\
                                (2*pow((orientations[neigh_patch_OFT_dir[Index_neg[Patch_Size*k+l]]*3]*\
                               orientations[k*3]+ orientations[neigh_patch_OFT_dir[Index_neg[Patch_Size*k+l]]*3+1]*\
                               orientations[k*3+1]+orientations[neigh_patch_OFT_dir[Index_neg[Patch_Size*k+l]]*3+2]*\
                               orientations[k*3+2]),2)-1)
            
        if Help_Max_var_pt[k] < 0:
            
            Help_Max_var_pt[k] = 0
        
        mean_value += Help_Max_var_pt[k]/Or_size
        
    ' Compute Standart deviation '    
    
    for k in range(Or_size):
        
        std_value += (mean_value -Help_Max_var_pt[k])*(mean_value -Help_Max_var_pt[k])/Or_size
 
         
    ' Return maximal value, index '
    
    for i in range(Or_size):       
        
        Help_Max_var_pt = &Help_Max_var[i]
        
        if Help_Max_var_pt[0] > max_value:
            
            max_value = Help_Max_var_pt[0]
            
            max_index = i          
            
    ' Copy result into LF_transform memoryview '
    
    Filter_transform[0] = max_value
    
    Filter_transform[1] = max_index
    
    Filter_transform[2] = mean_value
    
    Filter_transform[3] = std_value
    
    free(Help_Max_var)

