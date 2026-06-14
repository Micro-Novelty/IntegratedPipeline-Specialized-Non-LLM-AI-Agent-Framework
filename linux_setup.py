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
        'numpy',
        'scikit-learn',
        'hashlib',
        'os',
        'random',
        'sqlite3',
        'joblib',
        'ast',
        're',
        'json',
        'time',
        'datetime',
        'threading',
        'deque',
        'socket',
        'sys',
        'sklearn',
        'scipy',
        'collections',
        'pandas',
        'timedelta',
        'pickle',
        'ssl',

    ],
)
