import cupy as cp
import pathlib
import os

current_fpath = pathlib.Path(__file__).parent.absolute()
kernel_fname = "Energy_Radius.cu"
with open(os.path.join(current_fpath,kernel_fname), 'r') as f:
	kernel_src = f.read()

def energy_radius(data, radii, orientations, Rad_min,Rad_num, Num_Theta):
	"""
	Input

	data np.ndarray
		shape (X_dim, Y_dim, Z_dim)
	radii np.ndarray
		(num_radii)
	orientations np.ndarray
		(num_directions, 3)
	
	Output 
		Potential_Ret np.ndarray
		shape (X_dim, Y_dim, Z_dim, num_directions, num_radii)
	"""
	X_dim = data.shape[0]
	Y_dim = data.shape[1]
	Z_dim = data.shape[2]

	# partition (X_dim, Y_dim, Z_dim) grid into kron(grid_dim, blcok_dim)
	block_dim = tuple(3*[4])
	grid_dim = (X_dim//block_dim[0]+1, Y_dim//block_dim[1]+1, Z_dim//block_dim[2]+1)

	# move memory to gpu
	data_gpu = cp.array(data, dtype=cp.float32)
	radii_gpu = cp.array(radii, dtype=cp.float32)
	orientations_gpu = cp.array(orientations, dtype=cp.float32)

	# allocate output on gpu
	Potential_Ret = cp.zeros((*data.shape, radii.shape[0]),dtype=cp.float32)

	# compile kernel
	energy_radius_kernel = cp.RawKernel(kernel_src, 'energy_radius')

	# run kernel on GPU
	kernel_args = (data_gpu, radii_gpu, orientations_gpu, Potential_Ret,int(X_dim),\
			 int(Y_dim), int(Z_dim), int(orientations.shape[0]), int(radii.shape[0]),Rad_min,Rad_num,Num_Theta
	)
	energy_radius_kernel(*(grid_dim, block_dim), kernel_args)

	# move result to CPU Ram
	Potential_Ret = cp.asnumpy(Potential_Ret)
	return Potential_Ret

