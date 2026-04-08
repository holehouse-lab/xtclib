"""
xtclib – Read and write GROMACS XTC trajectory files.

This package provides a Cython/C implementation of the XTC compressed
coordinate format used by GROMACS for molecular dynamics trajectories.
It supports both reading (decompression) and writing (compression) of
XTC files, with output that is bit-identical to the reference GROMACS
implementation.

Key functions
-------------
read_xtc : Read an entire XTC file into NumPy arrays.
write_xtc : Write coordinates to an XTC file.
XTCReader : Iterator-based reader for frame-by-frame access.

Examples
--------
Read a trajectory:

>>> from xtclib import read_xtc
>>> data = read_xtc("trajectory.xtc")
>>> data["coords"].shape
(100, 1000, 3)

Write a trajectory:

>>> from xtclib import write_xtc
>>> write_xtc("output.xtc", data["coords"], time=data["time"])
"""

from .reader import XTCReader, read_xtc
from .writer import write_xtc
