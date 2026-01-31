"""Setup configuration for Wavecube library."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="wavecube",
    version="0.1.0-alpha",
    author="Alembic Project",
    author_email="",
    description="Multi-resolution wavetable matrix library for multimodal data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alembic/wavecube",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "gpu": [
            "cupy>=11.0.0",
        ],
        "full": [
            "cupy>=11.0.0",
            "opencv-python>=4.5.0",
            "h5py>=3.0.0",
            "Pillow>=9.0.0",
        ],
        "dev": [
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
