from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix
import numpy as np
from Cython_OFT import OFT_Graph_Utilities as cgraph
from Cython_OFT import OFT_Distance_Utilities_Cython
from Distance_Utilities import dist

def distance_stein(F, G):
    """Computes the matrix of Stein divergences between F and G.

    Parameters
    ----------
    F : array
        A (shape_F,s,s) array storing s-by-s HPD matrices.
    G : array
        A (shape_G,s,s) array storing s-by-s HPD matrices.

    Returns
    -------
    out : array
        A (shape_F,shape_G) array storing the Stein divergence D(F[i], G[j])
        in the component out[i,j].
    """

    # sanity checks and transformation of data:
    assert isinstance(F, np.ndarray)
    assert F.ndim >= 2
    assert G.ndim >= 2
    assert F.shape[-1] == F.shape[-2] == G.shape[-1] == G.shape[-2]
    s = F.shape[-1]
    shape_F = F.shape[:-2]
    F = F.reshape((-1, s, s))
    shape_G = G.shape[:-2]
    G = G.reshape((-1, s, s))

    out = np.empty((F.shape[0], G.shape[0]))
    if s == 3:
        out = Distance_Utilities_Cython.distance_stein_3x3(F, G, out)
    else:
        out = Distance_Utilities_Cython.distance_stein(F, G, out)
    out = out.reshape((*shape_F, *shape_G))
    return out

def adj_matrix_adaptive(F, p, sigma=1e0):
    """Computes the adjacency matrix of a grid graph with adaptive weights.

    The weights are given by w_ij ~ exp(-D(F_i,F_j)/sigma) up to normalization.
    """

    # sanity checks:
    assert isinstance(F, np.ndarray)
    assert F.ndim >= 2
    assert isinstance(p, (int, tuple))
    if isinstance(p, int):
        assert p > 0 and p % 2 == 1
    else:
        assert len(p) == 2 or len(p) == 3
        assert isinstance(p[0], int) and p[0] > 0 and p[0] % 2 == 1
        assert isinstance(p[1], int) and p[1] > 0 and p[1] % 2 == 1
        if len(p) == 3:
            assert isinstance(p[2], int) and p[2] > 0 and p[2] % 2 == 1
    assert sigma > 0.0
    # if isinstance(metric, str) and metric in ['Stein', 'Jeffrey', 'Frobenius']:
    #    assert F.ndim == 4
    #    assert F.shape[-1] == F.shape[-2]
    'Computation of 3D adaptive Adjucencematrix'
    if len(p) == 3:
        x, y, z, s, s = F.shape[:5]
        # m, n = F.shape[:2]
        F = F.reshape((x * y * z, s, s))
        if isinstance(p, tuple):
            mask = np.ones(p)
        data, indices, indptr = cgraph.adj_matrix_grid3d_uw((x, y, z), mask)
        for i in range(x * y * z):
            data[indptr[i]:indptr[i + 1]] *= \
                distance_stein(F[indices[indptr[i]:indptr[i + 1]]], F[i])
            # dist(F[indices[indptr[i]:indptr[i+1]]], F[i])
        data = np.exp(-data / sigma)
        data = cgraph.normalize_adj(data, indptr)
    else:
        z = 1
        x, y = F.shape[:3]
        # m, n = F.shape[:2]
        F = F.reshape((x * y))
        if isinstance(p, tuple):
            mask = np.ones(p)
        else:
            mask = np.ones((p, p))
        data, indices, indptr = cgraph.adj_matrix_grid2d_uw(x, y, mask)
        for i in range(x * y):
            data[indptr[i]:indptr[i + 1]] *= \
                distance_stein(F[indices[indptr[i]:indptr[i + 1]]], F[i])
            # dist(F[indices[indptr[i]:indptr[i+1]]], F[i])
        data = np.exp(-data / sigma)
        data = cgraph.normalize_adj(data, indptr)
    return csr_matrix((data, indices, indptr), shape=(x * y * z, x * y * z),
                      dtype=np.float64)


def adj_matrix_adaptive_Mod(F, p, sigma = 1e0):
    """Computes the adjacency matrix of a grid graph with adaptive weights.
    
    The weights are given by w_ij ~ exp(-D(F_i,F_j)/sigma) up to normalization.
    """
    
    # sanity checks:
    assert isinstance(F, np.ndarray)
    assert F.ndim >= 2
    assert isinstance(p, (int,tuple))
    if isinstance(p, int):
        assert p > 0 and p%2 == 1
    else:
        assert len(p) == 2 or len(p) == 3
        assert isinstance(p[0], int) and p[0] > 0 and p[0]%2 == 1
        assert isinstance(p[1], int) and p[1] > 0 and p[1]%2 == 1
        if len(p) == 3:
            assert isinstance(p[2],int) and p[2] >0 and p[2]%2 == 1
    assert sigma > 0.0

    'Computation of 3D adaptive Adjucencematrix'
    if len(p) == 3:
        x,y,z = F.shape[:3]
        #m, n = F.shape[:2]
        F = F.reshape((-1,1))
        if isinstance(p, tuple):
            mask = np.ones(p)
        data, indices, indptr = cgraph.adj_matrix_grid3d_uw((x, y, z), mask)
        for i in range(x*y*z):
            data[indptr[i]:indptr[i+1]] *= \
                dist(F[indices[indptr[i]:indptr[i+1]]], F[i])
            #dist(F[indices[indptr[i]:indptr[i+1]]], F[i])
        data = np.exp(-data/sigma)
        data = cgraph.normalize_adj(data, indptr)
    else:
        z = 1
        # For Desriptor
        x,y = F.shape[0:2]
        #x,y,s,s = F.shape[:5]
        #m, n = F.shape[:2]
        F = F.reshape(-1,1)
        if isinstance(p, tuple):
            mask = np.ones(p)
        else:
            mask = np.ones((p,p))
        data, indices, indptr = cgraph.adj_matrix_grid2d_uw(x,y, mask)
        for i in range(x*y):
            data[indptr[i]:indptr[i+1]] *= \
            dist(F[indices[indptr[i]:indptr[i+1]]], F[i])
                #distance_stein(F[indices[indptr[i]:indptr[i+1]]], F[i])
        data = np.exp(-data/sigma)
        data = cgraph.normalize_adj(data, indptr)
    return csr_matrix((data, indices, indptr), shape=(x*y*z, x*y*z), 
                      dtype=np.float64)

def adj_matrix(w, shape_img = None, normalize = True):
    """
    Returns the matrix A such that A*vec(I) = vec(correlate(I,w))
    for (m,n) images I.
    """
    
    # sanity checks:
    assert w.ndim == 2 or w.ndim == 3 or w.ndim == 4
    assert shape_img is not None or w.ndim == 4
    assert w.ndim != 2 or len(shape_img) == 2
    assert isinstance(normalize, bool)
    
    if w.ndim == 2:
        # assert np.all(w > 0.0) and np.any(w > 0.0)
        m,n = shape_img
        data, indices, indptr = ctools.adj_matrix_uniform(m, n, w)
    elif w.ndim == 3:
        x,y,z = w.shape[:3]
        #m, n = F.shape[:2]
        w = w.reshape((-1,1))
        if isinstance(shape_img, tuple):
            mask = np.ones(shape_img)
        data, indices, indptr = cgraph.adj_matrix_grid3d_uw((x, y, z), mask)
        #for i in range(x*y*z):
        #    data[indptr[i]:indptr[i+1]] *= \
        #        dist(F[indices[indptr[i]:indptr[i+1]]], F[i])
            #dist(F[indices[indptr[i]:indptr[i+1]]], F[i])
        #data = np.exp(-data/sigma)
        data = cgraph.normalize_adj(data, indptr)
    elif w.ndim == 4:
        m,n = w.shape[:2]
        data, indices, indptr = ctools.adj_matrix_nonuniform(w)
    #if normalize:
    #    data = ctools.normalize_adj(data, indptr)
    return csr_matrix((data, indices, indptr), shape=(x*y*z, x*y*z), 
                      dtype=np.float64)

' Return Entropy '

def Entropy(vec):
    if np.any(vec == 0):
        Entropy = -10
    else:
        n = len(vec)
        Entropy = -np.sum(vec * np.log(vec))
    return Entropy
