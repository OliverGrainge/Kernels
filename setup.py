import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.dirname(__file__))

setup(
    name='Ternarus',
    ext_modules=[
        CppExtension(
            name='vector_add_cpu',
            sources=[
                os.path.join('src/cpu/vectorAdd_cpu.cpp'),
                os.path.join('src/bindings.cpp')
            ],
            include_dirs=[os.path.join(project_root, 'include')],
            extra_compile_args=['-std=c++17']
        ),
        CUDAExtension(
            name='vector_add_gpu',
            sources=[
                os.path.join('src/gpu/vectorAdd_gpu.cu'),
                os.path.join('src/bindings.cpp')
            ],
            include_dirs=[os.path.join(project_root, 'include')],
            extra_compile_args={'cxx': ['-std=c++17'], 'nvcc': []}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
