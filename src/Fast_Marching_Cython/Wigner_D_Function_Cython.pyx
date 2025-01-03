'Import Modules'

import numpy as np
cimport numpy as cnp
import cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from openmp cimport omp_get_max_threads, omp_get_thread_num
from libc.math cimport signbit,exp,sin,cos, pi,sqrt,abs,fabs
from scipy.special.cython_special cimport sph_harm,eval_legendre
from cython.view cimport array as cvarray
from libc.string cimport memcpy

cdef extern from "math.h" nogil:
    int fmax(int,int)
    int fmin(int,int)
    double pow(double,int)
    
cdef extern from 'complex.h' nogil:
    double creal(complex)
    double complex I
    double cabs(complex)
    double carg(complex)
    double complex cpow(complex,int)
    
' Type Def '

ctypedef fused TYPE:
    double complex

' Transform Euler Angles to quaternion '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)

cdef void Euler2Quaternion_Cython(double alpha,double beta,double gamma, double complex*\
                                  quaternion):
    
    cdef double ca = cos(alpha * 0.5);
    cdef double sa = sin(alpha * 0.5);
    cdef double cb = cos(beta * 0.5);
    cdef double sb = sin(beta * 0.5);
    cdef double cc = cos(gamma * 0.5);
    cdef double sc = sin(gamma * 0.5);
    
    
    

    quaternion[0] = ca*cb*cc-sa*cb*sc+(sa*cb*cc+ca*cb*sc)*I
    quaternion[1] = ca*sb*sc-sa*sb*cc+(ca*sb*cc+sa*sb*sc)*I
    
' Get Wigner D coeficients for Basis expansion '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef inline double Wigner_D_Coef(int two_ell,int two_mp,int two_m) nogil:
    
    cdef double coef
    
    cdef int tworho_min  = fmax(0, two_mp - two_m)
                
    coef = sqrt(fac_cython((two_ell + two_m)//2) * fac_cython((two_ell - two_m)//2)\
                                        / (fac_cython((two_ell + two_mp)//2) * fac_cython((two_ell - two_mp)//2)))\
                                        * binomial_cython((two_ell + two_mp)//2, tworho_min//2)\
                                        * binomial_cython((two_ell - two_mp)//2, (two_ell - two_m - tworho_min)//2)

    return coef

' Input: Index L and 2D array with L+1 elements to be assubmled with indices in range [-L,L] '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef void Get_indices(int order,cnp.ndarray[int,ndim = 2] Array_indices):
    
    ' Intialization '
    
    cdef int i,k
    
    ' Create Array '
    
    s = 0
    
    for k in range(order):
        
        for i in range(-k,k+1):
            Array_indices[s,0] = k
            Array_indices[s,1] = i
            s += 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)        
   
cdef inline double fac_cython(int n) nogil:
    
    cdef int i
    cdef double ret = 1
    for i in range(1,n+1):
        ret = ret*i
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef inline double binomial_cython(int n, int k) nogil:
    
    ' Return Binomial Coeficient '

    if k < 0:
        return 0
    if k == 0:
        return 1
    if n < k:
        return 0
    
    cdef double p = 1
    cdef int N = min(k, n - k) + 1
    cdef int i
    
    for i in range(1, N):
        p = p * n
        p = p / i
        n = n - 1
    
    return p

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef inline double complex Wigner_D_Euler_cython(TYPE* quaternion,int L,int mp, int m) nogil:
    
    cdef  double epsilon            = 1e-10
    cdef  int two_ell               = L*2
    cdef  int two_mp                = mp*2
    cdef  int two_m                 = m*2
    cdef  double phase_a            = carg(quaternion[0])
    cdef  double radius_a           = cabs(quaternion[0])
    cdef  double phase_b            = carg(quaternion[1])
    cdef  double radius_b           = cabs(quaternion[1])
    cdef  double Abs_Ratio_Squared  = 0
    cdef  double Wigner_coef        = 0
    cdef  int tworho_min            = 0
    cdef  int tworho_max            = 0
    cdef double complex Prefactor   = 0.0+0.0*I
    cdef double coef                = 0
    cdef double Sum                 = 0.0
    cdef int two_N1                 = 0
    cdef int two_N2                 = 0
    cdef int two_M                  = 0
    cdef int l,k
    
    cdef double complex Wigner_D_Element = 0.0+0.0*I
    
    if radius_a <= epsilon:
        
        if two_mp != -two_m or fabs(two_mp) > two_ell or fabs(two_m)> two_ell:
            
            
            Wigner_D_Element = 0.0*I
        
        else:
        
            if (two_ell-two_m)%4 == 0:
                
                Wigner_D_Element = cpow(quaternion[1],two_m)
                
            else:
                
                Wigner_D_Element = -cpow(quaternion[1],two_m)
    
    elif radius_b < epsilon:
        
        if two_mp != two_m or fabs(two_mp) > two_ell or fabs(two_m) > two_ell:
            
            Wigner_D_Element = 0.0*I
            
        else:
            
            Wigner_D_Element = cpow(quaternion[0],two_m)
    
    elif radius_a < radius_b:
        
        Abs_Ratio_Squared = -(radius_a*radius_a)/(radius_b*radius_b)
        
        if fabs(two_mp) > two_ell or fabs(two_m) > two_ell:
            
            
            Wigner_D_Element = 0.0*I
            
        else:
            
            tworho_min  = fmax(0, -two_mp - two_m)
            
            coef = Wigner_D_Coef(two_ell,-two_mp,two_m)       
            
            Wigner_coef = coef*pow(radius_b,two_ell-(two_m+two_mp)/2-tworho_min)*\
                               pow(radius_a,(two_m+two_mp)/2+tworho_min) 
            Prefactor   = Wigner_coef*(cos(phase_b*(two_m-two_mp)/2+phase_a*(two_m+two_mp)/2)+I*sin\
                                     (phase_b*(two_m-two_mp)/2+phase_a*(two_m+two_mp)/2))
            
            if Prefactor == 0.0*I:
                
                Wigner_D_Element = 0.0*I
            
            else:
                
                if (two_ell -two_m -tworho_min) % 4 != 0:
                    
                    Prefactor = Prefactor*-1
                
                tworho_max = min(two_ell-two_mp,two_ell-two_m)
                
                two_N1 = two_ell - two_mp + 2
                
                two_N2 = two_ell - two_m + 2
                
                two_M  = two_m + two_mp
                
                Sum    = 1.0
                
                for l in range(tworho_max,tworho_min,-2):
                    
                    Sum = Sum*Abs_Ratio_Squared*((two_N1-l)*(two_N2-l))/(l*(two_M+l))
                    
                    Sum = Sum + 1
                    
                Wigner_D_Element = Prefactor*Sum
        
    else:
        
            
        Abs_Ratio_Squared = -radius_b*radius_b/(radius_a*radius_a)
            
        if fabs(two_mp) > two_ell or fabs(two_m) > two_ell:
            
                
            Wigner_D_Element = 0.0*I
                
        else:
                
            tworho_min = fmax(0,two_mp-two_m)
            
            
                
            coef = Wigner_D_Coef(two_ell,two_mp,two_m)
            
                            
            Wigner_coef = coef*pow(radius_a,two_ell-two_m/2+two_mp/2-tworho_min)*\
                               pow(radius_b,two_m/2-two_mp/2+tworho_min) 
            
                
            Prefactor = Wigner_coef*(cos(phase_a * (two_m+two_mp)/2+phase_b*(two_m-two_mp)/2)+\
                                         I*sin(phase_a * (two_m+two_mp)/2+phase_b*(two_m-two_mp)/2))
            
            
            if Prefactor == 0.0*I:
                    
                Wigner_D_Element = 0.0*I
                    
            else:
                    
                if (tworho_min % 4) != 0:
                        
                    Prefactor = Prefactor* -1
                        
                tworho_max = fmin(two_ell+two_mp,two_ell-two_m)
                    
                two_N1 = two_ell + two_mp + 2
                    
                two_N2 = two_ell - two_m + 2 
                    
                two_M = two_m -two_mp
                    
                Sum = 1.0 
                    
                for k in range(tworho_max,tworho_min,-2):
                    
                    Sum = Sum*Abs_Ratio_Squared*((two_N1-k)*(two_N2-k))/(k*(two_M+k))
                    
                    Sum = Sum + 1
                 
                Wigner_D_Element = Prefactor*Sum
            

    return Wigner_D_Element

#def void Spherical_Kernel_Rot(double alpha,double beta,double gamma, cnp.ndarray[double ,ndim = 2] angle_vector\
#                         ,double sigma, int Number_Grid, TYPE[:]* Wigner_Elements, int [:] indices):

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def Spherical_Kernel_Rot_Cython(double alpha,double beta,double gamma,cnp.ndarray[double ,ndim = 2] angle_vector\
                                ,double sigma,int order,int Number_Grid):
    
    ' Variable Initialization '
    
    cdef int k,l,n 
    
    cdef double complex* quaternion  =  <double complex*> malloc(sizeof(double complex)*2)
    
    cdef double complex* Quat_pt
    
    cdef cnp.ndarray[double complex,ndim = 1] Rotated_Wavelet = np.zeros(Number_Grid,dtype = np.complex128) 

    
    ' Set up quaternion '
    
    Quat_pt = &quaternion[0]
    
    Euler2Quaternion_Cython(alpha,beta,gamma,Quat_pt)
    
    for k in prange(Number_Grid,nogil = True,schedule = 'static'):
        
        for l in prange(order):
            
            for n in range(-l,l+1):
                
                Rotated_Wavelet[k] += Wigner_D_Euler_cython(Quat_pt, l, 0, n)*\
                                                        sph_harm(n,l,angle_vector[k,0], angle_vector[k,1])\
                                                        *(2*pi*eval_legendre(l,0.0)+I*(1-pow(-1,l))/2)*sqrt((2*l+1)/(4*pi))\
                                                                                *exp(-l*(l+1)*sigma) 
              
    free(quaternion)
    
    return Rotated_Wavelet
   
# Factorial in parallel using reduction --> prefered for computation of high factorials
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)

cpdef Fact_Reduction_cython(int n):
    
    ' Return Binomial Coeficient '

    cdef int k
    
    cdef int l = 1
    
    for k in prange(1,n+1,nogil = True):
    
        l *= k
        
    return l    