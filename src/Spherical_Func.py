import numpy as np
from scipy.special import lpmv,erf

' phi_2pi,theta_pi '

def polar_from_cartesian(cor_car):
    """ Embedded 3D unit vector from spherical polar coordinates.
    Parameters
    ----------
    x : array_like
        cartesian coordinates
    Returns
    -------
    phi, theta : float or numpy.array
        azimuthal and polar angle in radians.
    """
    
    'Normalize input vector '
    
    
    r = np.linalg.norm(cor_car,axis=1)
    x, y, z = cor_car[:,0],cor_car[:,1],cor_car[:,2]
    theta = np.arccos(z / r)
    phi = np.mod(np.arctan2(y, x), np.pi*2)
    return np.array([phi,theta,r])


def Spherical_Kernel_Rot(alpha,beta,angle_vector,sigma,order):
    
    Tranc_Array = np.arange(0,order)
    
    
    ' Set up array of indices for summation of Wigner functions '
    
    #indices = np.array([[ell,m] for ell in range(0,order) for m in range(-ell, ell+1)])
    
    
    Sum_ind = np.zeros((angle_vector.shape[0],order),dtype = "complex_")
    
    
    for angle in range(len(angle_vector)):
    
        ' Sum over big L '
    
        for Iterate in range(order):

            ' Sum over small l '

            c_0_ind        = np.zeros(2*Iterate+1,"complex_")
            Wigner_D_Euler = np.zeros(2*Iterate+1,"complex_")
            Spher_h        = np.zeros(2*Iterate+1,"complex_")

            for ind in range(-Iterate,Iterate+1,1):
                c_0_ind[ind]        = (2*np.pi*legendre(Iterate)(0)+1j*((1-(-1)**Iterate))/2)*np.sqrt((2*Iterate\
                                                +1)/(4*np.pi))*np.exp(-Iterate*(Iterate+1)*sigma)

                Wigner_D_Euler[ind] = sf.Wigner_D_element(0,beta,alpha,Iterate,0,ind)

                Spher_h[ind]        = sph_harm(ind,Iterate,angle_vector[angle][0],angle_vector[angle][1])
                
            Sum_ind[angle,Iterate] = np.sum(c_0_ind*Wigner_D_Euler*Spher_h,axis = 0)
        
        #print(str(angle)+'  Iteration finished  ')
        
        
    return np.sum(Sum_ind,axis = 1) 

    
def Spherical_Kernel(cor_car,shape_domain,order,sigma_0,sigma_p = 10,Legendre = True,Antisym = False):
    
    
    ' Set up array of indices for summation of spherical harmonics symmetric around z-axis '
    print(order)
    Tranc_Array = np.arange(0,order+1)
    
    
    ' For Funk transfor scaling by a Legendre polynomial '
    
    if Legendre:
        Array = legendre_pol(Tranc_Array,0)
        print('Funk')
    elif Antisym:
        print('Antisym')
        Array = 2*np.pi*legendre_pol(Tranc_Array,0)+1j*((1-(-1)**(Tranc_Array))/2)
    else:
        Array = np.ones(Tranc_Array.shape)
    order = Tranc_Array.shape[0]
    print(order)
    q = polar_from_cartesian(cor_car)[0:3,:]
    
    ' Initialize Output Array '
    
    L = np.zeros((order,q.shape[1]),dtype = complex)
    
    for k in range(q.shape[1]):
        L[:,k] = np.sqrt((2*Tranc_Array+1)/(4*np.pi))*np.exp(-(Tranc_Array+1)*Tranc_Array*sigma_0)\
                *np.array(list(map(lambda ord_val, degree:\
                                   sph_harm(ord_val,degree,q[0,k],q[1,k]),np.zeros(order),Tranc_Array)))#*Array
    
   
    ' Gaussian window for lower frequencies '
    
    
    L_1 = (1-gaussian(-(q[2,:]**2),sigma=sigma_p))*L*Array[:,None]
    
    return np.sum(L_1,axis=0)

def Radial_Erf(q,gamma,sigma_erf):
    
    ' Set Nyquist frequency '
    
    tau_N     = 3*sigma_erf/(1-gamma)
    
    print(tau_N)
    
    ' Set bounding point '
    
    sigma     = gamma*tau_N
    
    ' Initilize Array for Radial function '
    
    q_rad     = np.zeros(q.shape[1])
    
    
    for k in range(q.shape[1]):
        q_rad[k] = 1/2*(1-erf((q[2,k]-sigma)/sigma_erf))
    return q_rad

def legendre_pol(array_order,value):
    return np.array(list(map(lambda order: legendre(order)(value),array_order)))

def Spherical_Wigner_sum_m(order,theta,phi,alpha,beta):
    indices = np.array([m for m in range(-order, order+1)])
    return np.array(list((map(lambda ind: sph_harm(ind,order,theta,phi)*\
                    sf.Wigner_D_element(0,beta,alpha,order,0,ind),indices))),dtype = "complex_")



def coef_c_0_l(array_order,sigma):
    return np.array(list(map(lambda order:\
                   2*np.pi*(legendre(order)(0)+1j*(1-(-1)**order)/2)*np.sqrt((2*order+1)\
                    /(4*np.pi))*np.exp(-order*(order+1)*sigma),array_order)))



'Zernike Kernel'

from scipy.special import binom,factorial
from scipy.special import iv,gamma,poch,spherical_jn

def Get_p_max(alpha,beta):
    return (((1/2)*beta)/(alpha+(1/2)*beta))**(1/2)

def Real_Binomial_Coaf(alpha,p):
    Multiply_Array = np.prod(np.array([alpha-i for i in range(p)]))
    
    return  Multiply_Array/factorial(p)

def Zernike_Coaf_b(alpha,l,n,beta,p):
    return (Real_Binomial_Coaf((beta-l)/2,p)/(2*alpha+beta+l+2*p+3)*\
            (Real_Binomial_Coaf(1/2*(beta+l+1)+alpha+p,alpha+p)))

def Zernike_Coaf_flat_c(alpha,p_max):
    return np.array([1+(((1+alpha)**3)/(2*alpha))*p_max**4,-2*(((1+alpha)**3)\
                                                               /(2*alpha))*p_max**2,((1+alpha)**3)/(2*alpha)])
def Zernike_Coaf_flat_b(alpha,p,p_max,l):
    n = l+2*p
    c_array = Zernike_Coaf_flat_c(alpha,p_max)
    coaf_array = np.array(list(map(lambda ind : Zernike_Coaf_b(alpha,l,n,2+ind,p),np.arange(6)[::2])))
    return np.dot(c_array,coaf_array)

def Zernike_Coaf_c_0(n,l,sigma,alpha,p_max):
    
    p = int((n-l)/2)
    
    return ((legendre(l)(0)+(1-(-1)**l)/2)*np.sqrt((2*l+1)\
                    /(4*np.pi))*np.exp(-l*(l+1)*sigma))*Zernike_Coaf_flat_b(alpha,p,p_max,l)
 

def Func_S_alpha_n_l(value,alpha,n,l):
    
    p = (n-l)/2
    
    if value > 0 and type(alpha) != int:
        return ((2**alpha)*(-1)**p)*poch(p+1,alpha)*(np.pi/(2*value))**(1/2)*(iv(n+alpha+3/2,value)\
                                            /((value)**(alpha+1)))
    elif value == 0:
        if n == 0:
            return (sqrt(np.pi)*gamma(1+alpha))/(4*gamma(5/2+alpha))
        elif n != 0:
            return 0
    elif type(alpha) == int and value != 0:
        return ((2**alpha)*(-1)**p)*poch(p+1,alpha)*(spherical_jn(n+alpha+1,value)\
                                            /((value)**(alpha+1)))

def find_indeces(p):
    list_n = []
    list_l = []
    for n in range(int(p*p)):
        l = int(n-2*p)
        Value = int(n-l-2*p)
        print(Value)
        if Value == 0 and l > 0:
            print(n)
            list_n.append(n)
            list_l.append(l)
    return np.array(list_n),np.array(list_l)
