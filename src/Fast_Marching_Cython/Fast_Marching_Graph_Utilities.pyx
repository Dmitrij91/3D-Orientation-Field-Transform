import numpy as np
cimport numpy as cnp
cimport cython


ctypedef fused TYPE:
    float
    double
    long double



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def normalize_adj(cnp.ndarray[TYPE, ndim=1] data,
                  cnp.ndarray[cython.integral, ndim=1] indptr):
    """Normalize adj matrix so that rows sum up to 1."""

    cdef int i, j
    cdef TYPE s
    cdef int m = data.shape[0], n = indptr.shape[0]
    assert indptr[n-1] == m

    for i in range(n-1):
        s = 0.0
        for j in range(indptr[i], indptr[i+1]):
            s += data[j]
        for j in range(indptr[i], indptr[i+1]):
            data[j] /= s
    return data



@cython.wraparound(False)
@cython.boundscheck(False)
def adj_matrix_grid2d_uw(int m, int n, cnp.ndarray[TYPE, ndim=2] w):
    """Computes the adjacency matrix of a 2d grid with uniform weights.

    The output is a matrix A such that A*vec(I) = correlate(I,w) for
    (m,n) images I. Here, uniform weights means that the weight are the same
    for each neighborhood, while they can vary within the neighborhoods.
    """

    # sanity checks:
    assert w.shape[0]%2 == 1 and w.shape[1]%2 == 1
    assert m > 0 and n > 0

    # define constants:
    cdef int px = w.shape[0], py = w.shape[1]
    cdef int px2 = (px-1)//2, py2 = (py-1)//2
    cdef int num_entries = 1
    if m <= px2:
        num_entries *= m*m
    elif m >= px:
        num_entries *= (m-px+1)*px + (3*px-1)*(px-1)//4
    else:
        num_entries *= m*px - (px*px-1)//4
    if n <= py2:
        num_entries *= n*n
    elif m >= py:
        num_entries *= (n-py+1)*py + (3*py-1)*(py-1)//4
    else:
        num_entries *= n*py - (py*py-1)//4

    # initialize output variables:
    cdef cnp.ndarray[TYPE, ndim=1, negative_indices=False] data = \
        np.empty((num_entries,), dtype=w.dtype)
    cdef cnp.ndarray[int, ndim=1, negative_indices=False] indices = \
        np.empty((num_entries,), dtype=np.intc)
    cdef cnp.ndarray[int, ndim=1, negative_indices=False] indptr = \
        np.empty((m*n+1,), dtype=np.intc)
    cdef int i, j, k, l, a, b, ind_px, ind=0
    # Compute data of avrMatrix:
    indptr[0] = 0
    for i in range(m):
        for j in range(n):
            ind_px = 0
            for k in range(max(0, i-px2), min(m, i+px2+1)):
                for l in range(max(0, j-py2), min(n, j+py2+1)):
                    data[ind+ind_px] = w[k-i+px2, l-j+py2]
                    indices[ind+ind_px] = n*k+l
                    ind_px += 1
            ind += ind_px
            indptr[n*i+j+1] = ind
    #assert ind == num_entries
    return data, indices, indptr



@cython.wraparound(False)
@cython.boundscheck(False)
def adj_matrix_grid3d_uw((int,int,int) m, cnp.ndarray[TYPE, ndim=3] w):
    """Computes the adjacency matrix of a 3d grid with uniform weights.

    The output is a matrix A such that A*vec(I) = correlate(I,w) for
    (m,n) images I. Here, uniform weights means that the weight are the same
    for each neighborhood, while they can vary within the neighborhoods.
    """

    # sanity checks:
    assert w.shape[0]%2 == 1 and w.shape[1]%2 == 1 and w.shape[2]%2 == 1
    assert m[0] > 0 and m[1] > 0 and m[2] > 0

    #define constants:
    cdef (int,int,int) p = (w.shape[0], w.shape[1], w.shape[2])
    cdef (int,int,int) r = ((p[0]-1)//2, (p[1]-1)//2, (p[2]-1)//2) # = radius
    cdef long num_entries = 1

    # Computes the number of non-zero entries in the adj_matrix
    # (each dimension adds a factor):
    cdef int d
    # ctuples (m, p, r) are actually not made for being used in an for loop...
    for d in range(3):
        if m[d] <= r[d]:
            num_entries *= m[d] * m[d]
        elif m[d] >= p[d]:
            num_entries *= (m[d]-p[d]+1)*p[d] + (3*p[d]-1)*(p[d]-1)//4
        else:
            num_entries *= m[d] * p[d] - (p[d]*p[d]-1)//4

    # initialize output variables:
    cdef cnp.ndarray[TYPE, ndim=1, negative_indices=False] data = \
        np.empty((num_entries,), dtype=w.dtype)
    # use int64 to be safe with large pixel numbers:
    cdef cnp.ndarray[cnp.int64_t, ndim=1, negative_indices=False] indices = \
        np.empty((num_entries,), dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1, negative_indices=False] indptr = \
        np.empty((m[0]*m[1]*m[2]+1,), dtype=np.int64)
    cdef int i0, i1, i2, k0, k1, k2
    cdef long ind_px, ind=0

    # Compute entries of adj_matrix:
    indptr[0] = 0
    for i0 in range(m[0]):
        for i1 in range(m[1]):
            for i2 in range(m[2]):
                ind_px = 0
                for k0 in range(max(0, i0-r[0]), min(m[0], i0+r[0]+1)):
                    for k1 in range(max(0, i1-r[1]), min(m[1], i1+r[1]+1)):
                        for k2 in range(max(0, i2-r[2]), min(m[2], i2+r[2]+1)):
                            data[ind+ind_px] = \
                                w[k0-i0+r[0], k1-i1+r[1], k2-i2+r[2]]
                            indices[ind+ind_px] = m[1]*m[2]*k0 + m[2]*k1 + k2
                            ind_px += 1
                ind += ind_px
                indptr[m[1]*m[2]*i0+m[2]*i1+i2+1] = ind
    #assert ind == num_entries
    return data, indices, indptr



@cython.wraparound(False)
@cython.boundscheck(False)
def adj_matrix_grid2d_nuw(cnp.ndarray[TYPE, ndim=4] w):
    """Computes the adjacency matrix of a 2d grid with nonuniform weights.

    The first two dimensions of w are the dimensions of the grid. The last two
    dimensions of w are the weights within neighborhood. For example, if w has
    the dimensions (m,n,p,p) and (i,j) is a vertex of the m-by-n grid, then
    w[i,j,:,:] are the weights for the p-by-p neighborhood around this vertex.
    """

    # sanity checks:
    assert w.shape[2]%2 == 1 and w.shape[3]%2 == 1

    # define constants:
    cdef int m = w.shape[0], n = w.shape[1]
    cdef int px = w.shape[2], py = w.shape[3]
    cdef int px2 = (px-1)//2, py2 = (py-1)//2
    cdef int num_entries = 1
    if m <= px2:
        num_entries *= m*m
    elif m >= px:
        num_entries *= (m-px+1)*px + (3*px-1)*(px-1)//4
    else:
        num_entries *= m*px - (px*px-1)//4
    if n <= py2:
        num_entries *= n*n
    elif m >= py:
        num_entries *= (n-py+1)*py + (3*py-1)*(py-1)//4
    else:
        num_entries *= n*py - (py*py-1)//4

    # initialize output variables:
    cdef cnp.ndarray[TYPE, ndim=1, negative_indices=False] data = \
        np.empty((num_entries,), dtype=w.dtype)
    cdef cnp.ndarray[int, ndim=1, negative_indices=False] indices = \
        np.empty((num_entries,), dtype=np.intc)
    cdef cnp.ndarray[int, ndim=1, negative_indices=False] indptr = \
        np.empty((m*n+1,), dtype=np.intc)
    cdef int i, j, k, l, ind_px, ind = 0

    indptr[0] = 0
    for i in range(m):
        for j in range(n):
            ind_px = 0
            for k in range(px):
                if not 0 <= i + k - px2 < m:
                    continue
                for l in range(py):
                    if not 0 <= j + l - py2 < n:
                        continue
                    data[ind + ind_px] = w[i,j,k,l]
                    indices[ind + ind_px] = n * (i + k - px2) + (j + l - py2)
                    ind_px += 1
            ind += ind_px
            indptr[n*i+j+1] = ind
    #assert ind == num_entries
    return data, indices, indptr



@cython.wraparound(False)
@cython.boundscheck(False)
def adj_matrix_masked(cnp.ndarray[cnp.uint8_t, ndim=2, cast=True] mask,
                      cnp.ndarray[TYPE, ndim=2] w):
    cdef int mx = mask.shape[0], my = mask.shape[1]
    cdef int px = w.shape[0], py = w.shape[1]
    cdef int px2 = (px-1)//2, py2 = (py-1)//2
    cdef int num_ind = 0, num_entries = 0
    cdef cnp.ndarray[int, ndim=2] ind = np.empty((mx,my), dtype=np.intc)
    cdef cnp.ndarray[TYPE, ndim=1] data
    cdef cnp.ndarray[int, ndim=1] indices
    cdef cnp.ndarray[int, ndim=1] indptr
    cdef int i,j,k,l

    for i in range(mx):
        for j in range(my):
            if mask[i,j]:
                ind[i,j] = num_ind
                num_ind += 1
                for k in range(max(i-px2,0),min(i+px2+1,mx)):
                    for l in range(max(j-py2,0),min(j+py2+1,my)):
                        if mask[k,l]:
                            num_entries += 1
    data = np.empty((num_entries,), dtype=w.dtype)
    indices = np.empty((num_entries,), dtype=np.intc)
    indptr = np.empty((num_ind+1,), dtype=np.intc)

    num_ind = 0
    num_entries = 0
    indptr[0] = 0
    for i in range(mx):
        for j in range(my):
            if mask[i,j]:
                for k in range(max(i-px2,0),min(i+px2+1,mx)):
                    for l in range(max(j-py2,0),min(j+py2+1,my)):
                        if mask[k,l]:
                            data[num_entries] = w[k-i+px2,l-j+py2]
                            indices[num_entries] = ind[k,l]
                            num_entries += 1
                num_ind += 1
                indptr[num_ind] = num_entries
    return data, indices, indptr