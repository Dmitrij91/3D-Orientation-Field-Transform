from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name='OFT_Distance_Utilities_Cython',
        sources=['Distance_Utilities_Cython.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='OFT_Graph_Utilities',
        sources=['Fast_Marching_Graph_Utilities.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='Line_Filter_Transform_Cython',
        sources=['Line_Filter_Transform_Cython.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()])
setup(
      name = 'Accelerated_Routines',
      ext_modules = cythonize(extensions,
    compiler_directives={'language_level' : "3"})
)
