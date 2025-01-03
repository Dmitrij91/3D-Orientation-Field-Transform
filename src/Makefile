all:
	cd Fast_Marching_Cython; \
	python setup.py build_ext --inplace
	rm -f -r Fast_Marching_Cython/build
	rm -f Fast_Marching_Cython/Fast_*.c
	rm -f Fast_Marching_Cython/Distance*.c
	rm -f Fast_Marching_Cython/Wigner_D_Func*.c
	rm -f Fast_Marching_Cython/Line_Filter_Transform_Cython*.c
	rm -f Fast_Marching_Cython/Convolution*.c
	rm -f Fast_Marching_Cython/Orientation*.c
	rm -f Fast_Marching_Cython/Stochastic*.c
clean:
	rm -f Fast_Marching_Cython/Fast_*.so
	rm -f Fast_Marching_Cython/Distance*.so
	rm -f Fast_Marching_Cython/Wigner_D_Func*.so
	rm -f Fast_Marching_Cython/Line_Filter_Transform_Cython*.so
	rm -f Fast_Marching_Cython/Convolution*.so
	rm -f Fast_Marching_Cython/Orientation*.so
	rm -f Fast_Marching_Cython/Stochastic*.so
