from setuptools import setup, Extension
from setuptools_rust import RustExtension
from Cython.Build import cythonize
import numpy
import os

# path to your .pyx file — adjust if your module lives under a package directory
pyx_path = os.path.join("src", "AbstractIntegratedModule.pyx")
optimized_pyx_path = os.path.join("src", "AbstractOptimizedModules.pyx")

extensions = [
    Extension(
        "AbstractIntegratedModule",            # module name as imported in Python
        [pyx_path],
        # add include_dirs or libraries here, e.g. numpy.get_include()
        include_dirs=[numpy.get_include()],
        libraries=[],
    ),
    Extension(
        "AbstractOptimizedModules",  # module name as imported in Python
        [optimized_pyx_path],
        include_dirs=[numpy.get_include()],
        libraries=[],
    ),
]

setup(
    author="Micro-Novelty",
    version="0.7.3"
    author_email="hernikpuspita5@gmail.com",
    license="MIT",
    ext_modules=cythonize(extensions, language_level=3),
    rust_extensions=[
            RustExtension(
                "AbstractIntegratedModule.rust_module",   # compiled Rust fast-path
                path="rust_optimization_/Cargo.toml",
                binding="pyo3",
                optional=True,
            ),
    ],  
    python_requires="> 3.9",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
