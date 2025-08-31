from setuptools import setup
from distutils.extension import Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from Cython.Build import cythonize
import glob
import os
import numpy

project_name = "funcwm"
project_dir = os.getcwd()
numpy_include_dir = numpy.get_include()
eigen_headers = "-I/usr/include/eigen3"
headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")

ext_modules = []
cython_modules = []

lib_name = "cu3d"
lib_headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), project_dir, lib_name)
lib_sources = list(set(glob.glob(os.path.join(project_dir, lib_name, "*.cpp"))) - set(glob.glob(os.path.join(project_dir, lib_name, "main.cpp")))) + glob.glob(os.path.join(project_dir, lib_name, "*.cu"))

ext_modules.append(
    CUDAExtension(
        name=f'{project_name}.{lib_name}',
        sources=lib_sources,
        extra_compile_args = {
            "cxx": ["-O2", lib_headers,
                    headers,
                    ],
            "nvcc": ["-O2", lib_headers,
                     headers,
                     ]
        }
    )
)

lib_name = "cpu3d"
lib_headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), project_dir, lib_name)
lib_sources = list(set(glob.glob(os.path.join(project_dir, lib_name, "*.cpp"))) - set(glob.glob(os.path.join(project_dir, lib_name, "main.cpp"))))

ext_modules.append(
    CppExtension(
        name=f'{project_name}.{lib_name}',
        sources=lib_sources,
        extra_compile_args = {
            "cxx": ["-O2", lib_headers,
                    headers,
                    eigen_headers
                    ],
        }
    )
)

cython_modules.append(
    Extension(
        name=f'{project_name}.mcubes',
        sources=[
            f'{project_dir}/mcubes/mcubes.pyx',
            f'{project_dir}/mcubes/pywrapper.cpp',
            f'{project_dir}/mcubes/marchingcubes.cpp'
        ],
        language='c++',
        extra_compile_args=['-std=c++11'],
        include_dirs=[numpy_include_dir]
    )
)

cython_modules.append(
    Extension(
        name=f'{project_name}.dualcontour',
        sources=[
            f'{project_dir}/dualcontour/dualcontouring.pyx',
            f'{project_dir}/dualcontour/pywrapper.cpp',
            f'{project_dir}/dualcontour/vol/Contouring.cpp',
            f'{project_dir}/dualcontour/math/Math.cpp',
            f'{project_dir}/dualcontour/math/Solver.cpp',
        ],
        language='c++',
        extra_compile_args=['-std=c++11'],
        include_dirs=[numpy_include_dir]
    )
)

cython_modules.append(
    Extension(
        name=f'{project_name}.mise',
        sources=[
            f'{project_dir}/mise/mise.pyx'
        ],
        include_dirs=[numpy_include_dir]
    )
)

ext_modules.extend(cythonize(cython_modules))

setup(
    name = project_name,
    ext_modules = ext_modules,
    cmdclass = {
        'build_ext': BuildExtension
    }
)