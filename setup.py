from setuptools import setup, Extension
from setuptools_rust import RustExtension
from Cython.Build import cythonize
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
    ext_modules=cythonize(extensions, language_level=3),
    rust_extensions=[
            RustExtension(
                "AbstractIntegratedModule.rust_module",   # compiled Rust fast-path
                path="rust_optimization_/Cargo.toml",
                binding="pyo3",
                optional=True,
            ),
    ],        
)
