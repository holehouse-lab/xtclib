"""
Build configuration for xtclib.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "xtclib._xtc_core",
        sources=["xtclib/_xtc_core.pyx", "xtclib/xdr_xtc.c"],
        include_dirs=[np.get_include(), "xtclib"],
        language="c",
    ),
]

setup(
    name="xtclib",
    version="0.1.0",
    packages=["xtclib"],
    ext_modules=cythonize(extensions, compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
    }),
    install_requires=["numpy"],
    python_requires=">=3.8",
)
