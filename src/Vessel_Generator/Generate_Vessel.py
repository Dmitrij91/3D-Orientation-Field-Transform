import numpy as np
from scipy.linalg import null_space
from scipy.spatial.distance import cdist

def compute_batch_distances(Volume, Synthatic_Vessel_out, batch_size=10000):
    num_points = Volume.shape[0]
    distance_results = np.zeros(num_points)
    
    for start_idx in range(0, num_points, batch_size):
        end_idx = min(start_idx + batch_size, num_points)
        
        # Compute pairwise distances for the current batch
        batch = Volume[start_idx:end_idx, :]
        distance_results[start_idx:end_idx] = np.min(cdist(batch, Synthatic_Vessel_out), axis=1)
        
    return distance_results

def Rot_Mat_from_Rot_Axis_py(Rot_vec,angle):
    
    Rot_mat = np.zeros((3,3))
    
    Vec = Rot_vec/(np.linalg.norm(Rot_vec))
        
    ' Define Rotation Matrix '
    
    Rot_mat[0,0] = np.cos(angle)+(Vec[0]*Vec[0])*(1-np.cos(angle))
    Rot_mat[0,1] = Vec[0]*Vec[1]*(1-np.cos(angle))-Vec[2]*np.sin(angle)
    Rot_mat[0,2] = Vec[2]*Vec[0]*(1-np.cos(angle))+Vec[1]*np.sin(angle)
    Rot_mat[1,0] = Vec[1]*Vec[0]*(1-np.cos(angle))+Vec[2]*np.sin(angle)
    Rot_mat[1,1] = np.cos(angle)+Vec[1]*Vec[1]*(1-np.cos(angle))
    Rot_mat[1,2] = Vec[1]*Vec[2]*(1-np.cos(angle))-Vec[0]*np.sin(angle)
    Rot_mat[2,0] = Vec[2]*Vec[0]*(1-np.cos(angle))-Vec[1]*np.sin(angle)
    Rot_mat[2,1] = Vec[2]*Vec[1]*(1-np.cos(angle))+Vec[0]*np.sin(angle)
    Rot_mat[2,2] = np.cos(angle)+Vec[2]*Vec[2]*(1-np.cos(angle))
                  
    return Rot_mat


def Rx(theta):
    return np.array([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
  
def Ry(theta):
    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
def Rz(theta):
    return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])


import numpy as np

def generate_vessel_length(mean, std):
    return np.random.normal(mean, std)

# Assuming helper functions like Ry, Rz, Rot_Mat_from_Rot_Axis_py, generate_vessel_length, and compute_batch_distances are already defined elsewhere.

import numpy as np

# Assuming helper functions like Ry, Rz, Rot_Mat_from_Rot_Axis_py, generate_vessel_length, and compute_batch_distances are already defined elsewhere.

def Synthatic_Data_Random_Walk(Bifurcation_Num, Init_Dir, D_33, D_44, N, mean, std, x_length=200, y_length=200, z_length=200, depth=0, max_depth=3):
    """
    Recursive random walk to generate synthetic vessel networks.
    Each endpoint branches further by calling the function recursively.

    Parameters:
    - Bifurcation_Num: Number of bifurcations for each vessel.
    - Init_Dir: Initial direction of the vessel.
    - D_33, D_44: Parameters controlling the vessel length and propagation.
    - N: Number of segments in the vessel.
    - mean, std: Mean and standard deviation for vessel length generation.
    - x_length, y_length, z_length: Dimensions of the generated volume.
    - depth: Current depth of the recursion (default is 0).
    - max_depth: Maximum recursion depth (default is 3).

    Returns:
    - Volume_returned: Binary volume indicating the presence of vessels.
    - Volume_noisy: Noisy version of the volume.
    - Synthatic_Vessel_out: Generated vessel centerline.
    - Synthatic_Vessel_Filled: 3D coordinates of the filled vessel segments.
    """

    # Save tangent direction and spatial position
    Synthatic_Vessel = np.zeros((N, Bifurcation_Num, 2, 2, 3))
    Synthatic_Vessel_out = np.zeros((N * 2 * Bifurcation_Num, 3))
    Synthatic_Vessel_dir = np.zeros((N * 2 * Bifurcation_Num, 3))

    Synthatic_Vessel[0, 0, 0, 1, :] = Init_Dir
    Synthatic_Vessel[0, 0, 1, 1, :] = Init_Dir

    Rot_mat = np.zeros((3, 3))
    Euler_y_mat = np.zeros((3, 3))
    Euler_z_mat = np.zeros((3, 3))

    eps = 0.5

    # Vessel generation loop
    for l in range(Bifurcation_Num):
        for p in range(2):
            if l != 0:
                Synthatic_Vessel[0, l, p, 1, :] = np.random.rand(3)
                Synthatic_Vessel[0, l, p, 1, :] /= np.linalg.norm(Synthatic_Vessel[0, l, p, 1, :])
                Theta_Bifurcation = np.arccos(np.dot(Synthatic_Vessel[0, l, p, 1, :], Synthatic_Vessel[N - 1, l - 1, 0, 1, :]))

                while np.abs(Theta_Bifurcation) <= np.pi / 8 and np.abs(Theta_Bifurcation) >= np.pi / 4:
                    Synthatic_Vessel[0, l, p, 1, :] = np.random.rand(3)
                    Synthatic_Vessel[0, l, p, 1, :] /= np.linalg.norm(Synthatic_Vessel[0, l, p, 1, :])
                    Theta_Bifurcation = np.arccos(np.dot(Synthatic_Vessel[0, l, p, 1, :], Synthatic_Vessel[N - 1, l - 1, 0, 1, :]))

            Save_pos = Synthatic_Vessel[0, l, p, 0, :]

            for k in range(1, N):
                vessel_length = generate_vessel_length(mean, std)
                Cross_prod = np.cross(np.array([0, 0, 1]), Synthatic_Vessel[k - 1, l, p, 1, :])

                # Update Position
                Synthatic_Vessel[k, l, p, 0, :] = Save_pos + (D_33 / N) * eps * Synthatic_Vessel[k - 1, l, p, 1, :]
                Save_pos = Synthatic_Vessel[k, l, p, 0, :]

                beta = np.random.uniform(-1, 1)
                Euler_y_mat = Ry((D_44 / N) * beta)
                gamma = np.random.uniform(-0.5, 0.5)
                Euler_z_mat = Rz(gamma)

                theta = np.arcsin(np.linalg.norm(Cross_prod))
                Rot_mat = Rot_Mat_from_Rot_Axis_py(Cross_prod, theta)
                Synthatic_Vessel[k, l, p, 1, :] = Rot_mat @ Euler_z_mat @ Euler_y_mat @ Rot_mat.T @ Synthatic_Vessel[k - 1, l, p, 1, :]
                Synthatic_Vessel[k, l, p, 1, :] /= np.linalg.norm(Synthatic_Vessel[k, l, p, 1, :])

    # Connect random vessel lines
    Synthatic_Vessel_out[0:N, :] = Synthatic_Vessel[:, 0, 0, 0, :]
    Synthatic_Vessel_out[N:2 * N, :] = Synthatic_Vessel_out[N - 1, :] + Synthatic_Vessel[:, 0, 1, 0, :]
    Synthatic_Vessel_dir[0:N, :] = Synthatic_Vessel[:, 0, 0, 1, :]
    Synthatic_Vessel_dir[N:2 * N, :] = Synthatic_Vessel[:, 0, 1, 1, :]

    # Generate vessel centerline
    for l in range(1, Bifurcation_Num):
        Synthatic_Vessel_out[2 * N * l:N * l * 2 + N, :] = Synthatic_Vessel_out[2 * (l - 1) * N + N - 1, :][None, :] + Synthatic_Vessel[:, l, 0, 0, :]
        Synthatic_Vessel_out[(2 * N * l + N):(N * l * 2 + 2 * N), :] = Synthatic_Vessel_out[2 * (l - 1) * N + N - 1, :][None, :] + Synthatic_Vessel[:, l, 1, 0, :]
        Synthatic_Vessel_dir[2 * N * l:N * l * 2 + N, :] = Synthatic_Vessel[:, l, 0, 1, :]
        Synthatic_Vessel_dir[(2 * N * l + N):(N * l * 2 + 2 * N), :] = Synthatic_Vessel[:, l, 1, 1, :]

    # Base case: if max depth is reached, return current data
    if depth >= max_depth:
        return Synthatic_Vessel_out, Synthatic_Vessel_Filled, Synthatic_Vessel_dir

    # Recursively generate vessels from each endpoint
    for k in range(N * 2 * Bifurcation_Num):
        new_init_dir = Synthatic_Vessel_dir[k, :]
        new_position = Synthatic_Vessel_out[k, :]
        new_Bifurcation_Num = 2  # Each endpoint generates two new vessels
        Synthatic_Data_Random_Walk(Bifurcation_Num=new_Bifurcation_Num,
                                    Init_Dir=new_init_dir,
                                    D_33=D_33,
                                    D_44=D_44,
                                    N=N,
                                    mean=mean,
                                    std=std,
                                    x_length=x_length,
                                    y_length=y_length,
                                    z_length=z_length,
                                    depth=depth + 1,
                                    max_depth=max_depth)  # Increase depth for recursion

    # Generate volume and noisy volume as before (same logic)
    Volume = np.zeros((x_length, y_length, z_length, 3))
    alpha = np.linspace(0, 2 * np.pi, 40)
    radius = np.linspace(0, (D_33 / N) * eps, 10)
    Synthatic_Vessel_Filled = np.zeros((N * 2 * Bifurcation_Num - N, 10, 40, 3))

    for k in range(N * 2 * Bifurcation_Num - N):
        for l in range(10):
            for m in range(40):
                Vec = Synthatic_Vessel_dir[k, :] / np.linalg.norm(Synthatic_Vessel_dir[k, :])
                Rot = Rot_Mat_from_Rot_Axis_py(Vec, alpha[m])
                Orth_Vec = null_space(np.matrix(Vec))[:, 0]
                Synthatic_Vessel_Filled[k, l, m, :] = radius[l] * Rot @ Orth_Vec + Synthatic_Vessel_out[k, :]

    # Generate volume
    ind = np.indices(Volume.shape[:-1]) - np.array([25, 25, 25])[:, None, None, None]
    filter_funcs = []
    filter_funcs.append(lambda I, res=ind[2][::-1]: (2 * ind[2][::-1] / Volume.shape[0]))
    filter_funcs.append(lambda I, res=ind[1][::-1]: (2 * ind[1][::-1] / Volume.shape[0]))
    filter_funcs.append(lambda I, res=ind[0][::-1]: (2 * ind[0][::-1] / Volume.shape[0]))

    for s, filter_ in enumerate(filter_funcs):
        Volume[:, :, :, s] = filter_(Volume)

    dim = Volume.shape[0]
    Volume = Volume.reshape(-1, 3)
    Volume_returned = np.zeros(Volume.shape[0])
    Volume_returned[np.where(compute_batch_distances(Volume, Synthatic_Vessel_out) < 0.1)] = 1
    Volume_noisy = Volume_returned.copy()
    Volume_noisy = (Volume_noisy * (1 - np.round(np.random.normal(mean, std, dim ** 3)).astype(bool).astype(int))).reshape((dim, dim, dim))
    Volume_noisy = Volume_noisy + np.random.normal(0, 0.02, (dim, dim, dim))
    Volume_returned = Volume_returned.reshape(dim, dim, dim)

    return Volume_returned, Volume_noisy, Synthatic_Vessel_out, Synthatic_Vessel_Filled.reshape(-1, 3)
