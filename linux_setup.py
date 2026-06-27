# for compilation only in this repo.
# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import platform

# Detect architecture for ARM64-specific optimization
is_arm64 = platform.machine() in ('aarch64', 'arm64')

# List all your .pyx files
extensions = [
    Extension(
        "AbstractIntegratedModule",  # Module name (import AbstractIntegratedModule)
        sources=["src/AbstractIntegratedModule.pyx"],  # Your Cython source file
        include_dirs=[np.get_include()],
        extra_compile_args=['-O2', '-march=native'],  # Optimizations
        extra_link_args=[]
    ),
    Extension(
        "AbstractOptimizedModules",  # Module name (import AbstractOptimizedModules)
        sources=["src/AbstractOptimizedModules.pyx"],  # ARM64 optimized Cython source file
        include_dirs=[np.get_include()],
        extra_compile_args=['-O2', '-march=native'] if is_arm64 else ['-O2', '-march=native'],  # ARM64 gets higher optimization
        extra_link_args=[]
    ),
]

setup(
    name="AbstractIntegratedModule",
    description="AbstractIntegratedModule - A Cython implementation of the AbstractIntegratedModule and AbstractOptimizedModules",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,  # Disable for speed
            'initializedcheck': False,  # Disable for speed
            'nonecheck': False,  # Disable for speed
        }
    ),
    include_dirs=[np.get_include()],
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "joblib>=1.1.0",
        "cryptography>=41.0.0",
        "aiohttp>=3.8.0",
        "psutil>=5.9.0",
    ],
)
