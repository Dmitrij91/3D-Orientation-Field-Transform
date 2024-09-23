all:
	cd ../src/Cython_OFT; \
	python setup.py build_ext --inplace
	rm -f -r Cython_OFT/build
	rm -f Cython_OFT/Line_Filter_Transform_Cython*.c
	rm -f Cython_OFT/OFT*.c
clean:
	rm -f Cython_OFT/Line_Filter_Transform_Cython*.so
	rm -f Cython_OFT/OFT*.so
