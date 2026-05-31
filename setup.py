# setup.py
from setuptools import setup
import os

# Read README for long description
readme_path = "README.md"
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Advanced Integrated Non-LLM AI Module - Backend Framework for Non-LLM AI Agent Framework"

setup(
    name="AbstractIntegratedModule",
    version="0.1.7", # matched PIP version, not binary version.
    description="Advanced Integrated AI Module - Multi-agent P2P inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Micro-Novelty",
    author_email="hernikpuspita5@gmail.com",
    license="MIT",
    
    # single file module
    py_modules=["AbstractIntegratedModule"],
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "joblib>=1.1.0",
        "cryptography>=41.0.0",
        "aiohttp>=3.8.0",
        "psutil>=5.9.0",
    ],
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    
    
    # Include package data (if any)
    include_package_data=True,
    
    # URLs
    url="https://github.com/Micro-Novelty/IntegratedPipeline-Specialized-Non-LLM-AI-Agent-Framework",
    project_urls={
        "Bug Reports": "https://github.com/Micro-Novelty/IntegratedPipeline-Specialized-Non-LLM-AI-Agent-Framework/issues",
        "Source": "https://github.com/Micro-Novelty/IntegratedPipeline-Specialized-Non-LLM-AI-Agent-Framework",
        "Documentation": "https://github.com/Micro-Novelty/IntegratedPipeline-Specialized-Non-LLM-AI-Agent-Framework#readme",
    },
)
