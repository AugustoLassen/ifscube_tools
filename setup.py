# setup.py
from setuptools import setup, find_packages

setup(
    name="ifscube_tools",
    version="0.1.1",
    packages=find_packages(),  # automatically finds ifscube_tools
    install_requires=[
        "numpy",
        "pandas",
        "astropy",
        "pyneb",
        "tqdm",
        "uncertainties",
    ],
    description="Tools to handle IFSCUBE data cubes and 1D spectra",
    author="Your Name",
    author_email="augusto.lassen@gmail.com",
    python_requires=">=3.8",
)
