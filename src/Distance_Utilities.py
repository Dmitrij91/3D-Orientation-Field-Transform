from scipy.spatial.distance import euclidean
import numpy as np

def dist(X, Y):

    """Computes the matrix out[i,j] = 0.5 * |X[i] - Y[j]|^2."""

    # sanity checks and transformation of data:
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert X.ndim >= 1
    assert Y.ndim >= 1
    shape_X = X.shape
    X = X.reshape((-1, shape_X[-1]))
    shape_Y = Y.shape
    Y = Y.reshape((-1, shape_Y[-1]))

    out = - 2.0 * np.real(np.conj(X).dot(Y.T))
    out += np.sum(np.abs(X) ** 2, axis=1)[:, None]
    out += np.sum(np.abs(Y) ** 2, axis=1)[None, :]
    out = out.reshape((*shape_X[:-1], *shape_Y[:-1]))
    return 0.5 * out


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
        out = cdistance.distance_stein_3x3(F, G, out)
    else:
        out = cdistance.distance_stein(F, G, out)
    out = out.reshape((*shape_F, *shape_G))
    return out

############################################################
######' Input : ' \                                  #######
######' Data shape:     N * s * s ' \                #######
######' N_f = Number of data points ' \              #######
######' N_p = Number of prototypes'\                 #######
######' s = feature dimension pf SPD matrix '\       #######
######' Output : N_f * N_p Distance Matrix '\        #######
############################################################
def distance_Riemmann(data, prototype):

    """Computes the matrix of Riemmannian distances between data and prototypes."""

    # sanity checks:

    assert data.shape[1] == data.shape[2]
    assert prototype.shape[1] == prototype.shape[2]
    assert data.shape[1] == prototype.shape[1]

    ' Descriptor shape '

    d = data.shape[1]

    ' Datasize '

    m = data.shape[0]

    ' Number of Prototypes '

    K = prototype.shape[0]

    ' Initilize Memory for Output of size m times K  '

    out = np.empty((m, K))

    'Iterate over Nodes in parallel using omp_get_thread_num() for current iterator index '

    for i in range(m):

        ' Copy Covariance matrices from data into shared memory by filling the rowes of size d times d '

        ' Iterate over Prototypes '

        for j in range(K):
            out[i, j] = Riemmannian_Distance_Cython(data[i, :, :], prototype[j, :, :])

    return np.asarray(out)
