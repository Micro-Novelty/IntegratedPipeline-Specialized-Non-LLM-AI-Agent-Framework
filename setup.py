from setuptools import setup
from setuptools.dist import Distribution
import os
import sys

class BinaryDistribution(Distribution):
    """Distribution class for binary extensions"""
    def has_ext_modules(self):
        return True
    
    def is_pure(self):
        return False

# Read README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Detect platform for binary selection (optional)
is_windows = sys.platform == "win32"
is_linux = sys.platform.startswith("linux")
is_arm64 = "aarch64" in os.uname().machine if hasattr(os, 'uname') else False

setup(
    name="AbstractIntegratedModule",
    version="0.5.0",
    description="Framework for Advanced Integrated Non-LLM AI Module library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Micro-Novelty",
    author_email="hernikpuspita5@gmail.com",
    license="MIT",
    
    # Module configuration
    py_modules=["AbstractIntegratedModule"],
    
    # Include binary files
    package_data={
        "": [
            "*.pyd",
            "*.so",
            "*.dll",
        ],
    },
    
    # Platform-specific binary inclusion
    data_files=[
        ("bin", [
            "AbstractIntegratedModule.cp313-win_amd64.pyd",
            "AbstractIntegratedModule.cpython-310-aarch64-linux-gnu.so",
            "AbstractIntegratedModule.cpython-312-x86_64-linux-gnu.so",
        ]),
    ],
    
    distclass=BinaryDistribution,
    python_requires=">=3.9,<3.14",
    
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "joblib>=1.1.0",
        "cryptography>=41.0.0",
        "aiohttp>=3.8.0",
        "psutil>=5.9.0",
    ],
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ],
    
    # For binary distributions, these are important
    zip_safe=False,
    include_package_data=True,
)