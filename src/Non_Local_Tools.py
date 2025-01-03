import numpy as np
from Fast_Marching_Cython.Fast_Marching_Non_Local_Tools import Gauss_Jacobi_Iteration_Cython, Nonlocal_Gradient_Cython, Gamma_Weights_Cython
from scipy import ndimage
from matplotlib import pyplot as plt

' Nonlocal discrete Diffusion on graphs '

' Input f_0 '
' sparse adjucent matrix with nonlocal means distances passed as '
' data,indices, indptr '

beta = 1e-8


def Non_Local_Diffusion(f_0, patch_weights, indices, indptr, shape, p=0.001, eps=1e-2, Max_Iter=5):
    Iter = 0

    sigma_t = ndimage.variance(f_0.reshape(shape))

    print(sigma_t)

    f_iter = f_0.copy()

    f_iter = Gauss_Jacobi_Iteration_Cython(f_0, patch_weights, indices, indptr, f_iter, sigma_t, p, Iter, 11, \
                                           beta)

    while Check_Convergence(f_iter, f_0) > eps and Iter < Max_Iter:
        Iter += 1
        sigma_t = ndimage.variance(f_iter.reshape(shape))
        print(sigma_t)
        f_iter = Gauss_Jacobi_Iteration_Cython(f_0, patch_weights, indices, indptr, f_iter, sigma_t, p, Iter, 11, beta)
        print(str(Iter) + '    Iteration of nonlocal diffussion ')
        plt.figure(Iter)
        plt.imshow(f_iter.reshape(shape))

    return f_iter


' Computes nonlocal gradient from patch and patch_weights (nonlocal means) '
' Both inputs must coinced with respect to adj_matrix shape '

# def Nonlocal_Gradient(patch_data,patch_weights):
#
#    Length_Patch = patch_data.shape[0]
#    Output       = 0
#
#
#    for k in range(Length_Patch):
#            Output += 0.15 + patch_weights[k]*(patch_data[k]-patch_data[0])**2
#
#
#    return Output**(1/2)


' Returns Total Variation Energy '


def Return_Energy(data, weights, indices, indptr, p):
    ' Length of Processed Image '

    N = len(indptr)

    Reg_Energy = 0

    for k in range(N - 1):
        Reg_Energy += 1 / p * (Nonlocal_Gradient_Cython(data[indptr[k]:indptr[k + 1]], weights[indptr[k]:indptr[k + 1]] \
                                                        , beta)) ** (p / 2)

    return Reg_Energy


' One step of Gauss Jacobi iteration step '


def Gauss_Jacobi_Iteration(f_0, weights, indices, indptr, f_iter, sigma_t, p, Iter):
    ' Intialization '

    data_gamma = Gamma_Weights_Cython(weights, indptr, indices, f_iter, p, beta, 11)

    print(' Weights_Computed ')

    Bool = np.any(np.isnan(data_gamma) == True)

    print(str(Bool))

    N = len(f_iter)
    if Iter == 0:
        lambda_reg = 1 / sigma_t ** 2
    else:
        lambda_reg = Update_Lambda(f_iter, f_0, sigma_t, data_gamma,indices, indptr)

    print(str(lambda_reg) + 'Lambda_Reg')

    print('Gauss Seidel Iteration')

    for k in range(N - 1):
        f_iter[k] = (lambda_reg * f_0[k] + np.sum(data_gamma[indptr[k]:indptr[k + 1]] * \
                                                  f_iter[indices[indptr[k]:indptr[k + 1]]], axis=0)) \
                    / (lambda_reg + np.sum(data_gamma[indptr[k]:indptr[k + 1]], axis=0))

    Bool = np.any(np.isnan(f_iter) == True)

    print(str(Bool) + ' Check f_iter')

    return f_iter


' Nonlocal P Laplacian '


def Update_Lambda(f_iter, f_0, sigma_t, data_gamma,indices, indptr):
    ' Initialization '

    N = len(f_iter)
    print(N)
    Lambda_t = 0

    for k in range(N - 1):
        Patch_sum = np.sum(data_gamma[indptr[k]:indptr[k + 1]] * (-f_iter[indices[indptr[k]:indptr[k + 1]]] \
                                                                  + f_iter[indices[indptr[k]]]), axis=0)

        Lambda_t += (1 / ((sigma_t ** 2) * N)) * (f_iter[k] - f_0[k]) * Patch_sum

    return Lambda_t

def Gamma_Weights(data_weights, indptr, indices, f, p):
    data_gamma = np.zeros_like(data_weights)
    N = len(indptr)

    print('Compute Gamma Weights')

    for k in range(N - 1):

        patch_pixel = f[indices[indptr[k]:indptr[k + 1]]]
        patch_weights_pixel = data_weights[indptr[k]:indptr[k + 1]]
        grad_value = Nonlocal_Gradient_Cython(patch_pixel, patch_weights_pixel, beta)

        patch_indices = indices[indptr[k]:indptr[k + 1]]
        # print(patch_indices)

        for ind_k, l in enumerate(patch_indices):
            if indptr[l + 1] < indptr[-1]:
                patch_data = f[indices[indptr[l]:indptr[l + 1]]]
                patch_weights = data_weights[indptr[l]:indptr[l + 1]]
                Length_Patch = len(patch_weights)
                for ind in range(Length_Patch):
                    # print(indptr[l]+ind)
                    data_gamma[indptr[l] + ind] = data_weights[indptr[k] + ind_k] * (
                                Nonlocal_Gradient_Cython(patch_data, \
                                                         patch_weights, beta) ** (p - 2) + grad_value ** (p - 2))
        # data_gamma[indptr[k]:indptr[k+1]]  = data_gamma[indptr[k]:indptr[k+1]]/np.sum(data_gamma[indptr[k]:\
        #                                                                                    indptr[k+1]])
    print('Finisched')
    return data_gamma


def Check_Convergence(f_up, f_old):
    return np.linalg.norm(f_up - f_old) / np.linalg.norm(f_up)


' According to Paper by Olver Lezoray:  '
'Parameterless discrete regularization on graphs for color image filtering'


def Nonlocal_Sigma_Par(f_0):
    ' Can be modified to patches '

    win_rows, win_cols = 13, 13
    win_mean = ndimage.uniform_filter(f_0, (win_rows, win_cols))
    win_sqr_mean = ndimage.uniform_filter(f_0 ** 2, (win_rows, win_cols))
    win_var = win_sqr_mean - win_mean ** 2

    return win_var


