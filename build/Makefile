all:
	cd ../src/Cython_OFT; \
	python setup.py build_ext --inplace
	rm -f -r Cython_OFT/build
	rm -f Cython_OFT/Line_Filter_Transform_Cython*.c
clean:
	rm -f Cython_OFT/Line_Filter_Transform_Cython*.so
