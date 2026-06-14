# for compilation only in this repo.
# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# List all your .pyx files
extensions = [
    Extension(
        "AbstractIntegratedModule",  
        sources=["AbstractIntegratedModule.pyx"],  # Cython source file
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3', '-march=native'],  # Optimizations
        extra_link_args=[]
    ),
    # Add more extensions if you have multiple .pyx files
    # Extension("transformer", sources=["transformer.pyx"]),
]

setup(
    name="AbstractIntegratedModule",
    version="1.0",
    description="AbstractIntegratedModule - A Cython implementation of the AbstractIntegratedModule",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,  # Disable for speed
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
