from setuptools import setup, Extension
from setuptools_rust import RustExtension, Binding
from Cython.Build import cythonize
import numpy
import os

pyx_path = os.path.join("src", "AbstractIntegratedModule.pyx")
optimized_pyx_path = os.path.join("src", "AbstractOptimizedModules.pyx")

_compile_args = [
    "-O2",
    "-fno-strict-aliasing",   # critical for Cython-generated C
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
]

_cython_directives = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "nonecheck": False,
    "initializedcheck": False,  # skip memoryview init checks
}

extensions = [
    Extension(
        "AbstractIntegratedModule",
        [pyx_path],
        include_dirs=[numpy.get_include()],
        extra_compile_args=_compile_args,
        extra_link_args=["-O2"],
        libraries=[],
    ),
    Extension(
        "AbstractOptimizedModules",
        [optimized_pyx_path],
        include_dirs=[numpy.get_include()],
        extra_compile_args=_compile_args,
        extra_link_args=["-O2"],
        libraries=[],
    ),
]

setup(
    author="Micro-Novelty",
    version="0.7.8",
    author_email="hernikpuspita5@gmail.com",
    license="MIT",
    ext_modules=cythonize(
        extensions,
        compiler_directives=_cython_directives,
        nthreads=os.cpu_count(),   # parallel Cython compilation across cores
    ),
    rust_extensions=[
        RustExtension(
            "AbstractIntegratedModule.rust_optimization_",
            path="rust_optimization_/Cargo.toml",
            binding=Binding.PyO3,   # use enum instead of string — safer across versions
            optional=True,
            debug=False,            # always release mode, never debug symbols in wheels
            features=["extension-module"],  # explicit PyO3 feature flag
        ),
    ],
    python_requires=">=3.10",   # dropped 3.9 support based on wheel targets
    zip_safe=False,             # required for binary extensions — prevents zip import issues
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
