from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name='Distance_Utilities_Cython',
        sources=['Distance_Utilities_Cython.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='Fast_Marching_Graph_Utilities',
        sources=['Fast_Marching_Graph_Utilities.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='Wigner_D_Function_Cython',
        sources=['Wigner_D_Function_Cython.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='Line_Filter_Transform_Cython',
        sources=['Line_Filter_Transform_Cython.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
     Extension(
        name='Convolution_3D',
        sources=['Convolution_3D.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='Orientation_Score_Enhancment_Filter',
        sources=['Orientation_Score_Enhancment_Filter.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]),
    Extension(
        name='Stochastic_Kernel_Cython',
        sources=['Stochastic_Kernel_Cython.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='Fast_Marching_Energy',
        sources=['Fast_Marching_Energy.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        include_dirs=[numpy.get_include()]),
]

setup(
      name = 'Optimized methods',
      ext_modules = cythonize(extensions,
    compiler_directives={'language_level' : "3"})
)