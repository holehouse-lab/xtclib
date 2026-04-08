# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython wrapper around the C XTC (de)compression routines.
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

np.import_array()

cdef extern from "xdr_xtc.h":
    int xtc_3d_decompress(const unsigned char *buf, int buf_len,
                           int natoms, float precision,
                           const int minint[3], const int maxint[3],
                           int smallidx,
                           float *out_coords)

    int xtc_3d_compress(const float *coords, int natoms, float precision,
                         unsigned char *out_buf, int out_buf_cap, int *out_len,
                         int out_minint[3], int out_maxint[3], int *out_smallidx)


def decompress_coords(const unsigned char[:] compressed not None,
                      int natoms,
                      int minint_x, int minint_y, int minint_z,
                      int maxint_x, int maxint_y, int maxint_z,
                      int smallidx,
                      float precision):
    """
    Decompress XTC-compressed coordinate data for a single frame.

    This is a low-level function called internally by `XTCReader`.
    Most users should use `read_xtc` or `XTCReader` instead.

    Parameters
    ----------
    compressed : bytes or bytearray
        The raw compressed payload extracted from the XTC frame, after
        the frame header (magic, natoms, step, time, box, precision,
        minint, maxint, smallidx, and size fields have been read).
    natoms : int
        Number of atoms.  Must match the value in the frame header.
    minint_x, minint_y, minint_z : int
        Minimum integer coordinate values along each axis, read from
        the frame header.  These define the lower bound of the
        coordinate range.
    maxint_x, maxint_y, maxint_z : int
        Maximum integer coordinate values along each axis, read from
        the frame header.  Together with ``minint_*`` they define the
        full-range encoding width.
    smallidx : int
        Index into the XTC magic-integer table that sets the initial
        "small" encoding range for run-length compressed deltas.
        Read directly from the frame header.
    precision : float
        The precision factor stored in the frame header (typically
        1000.0).  Decompressed integer coordinates are divided by this
        value to recover floating-point positions in nm.

    Returns
    -------
    coords : ndarray, shape (natoms, 3), dtype float32
        Decompressed Cartesian coordinates in nm.

    Raises
    ------
    RuntimeError
        If the C decompression routine signals an error (e.g. corrupt
        data or buffer underrun).

    Notes
    -----
    The header parameters (minint, maxint, smallidx, precision) are
    *not* validated against each other here; the caller is responsible
    for ensuring they match the compressed payload.

    See Also
    --------
    compress_coords : The inverse operation.
    read_xtc : High-level reader that calls this automatically.
    """
    cdef int minint[3]
    cdef int maxint[3]
    minint[0] = minint_x; minint[1] = minint_y; minint[2] = minint_z
    maxint[0] = maxint_x; maxint[1] = maxint_y; maxint[2] = maxint_z

    cdef np.ndarray[np.float32_t, ndim=2] coords = np.empty((natoms, 3), dtype=np.float32)

    cdef int rc = xtc_3d_decompress(
        &compressed[0], <int>compressed.shape[0],
        natoms, precision,
        minint, maxint, smallidx,
        <float *>coords.data)

    if rc != 0:
        raise RuntimeError("XTC decompression failed")

    return coords


def compress_coords(np.ndarray[np.float32_t, ndim=2] coords not None,
                    float precision=1000.0):
    """
    Compress coordinate data using the XTC algorithm.

    This is a low-level function called internally by `write_xtc`.
    Most users should use `write_xtc` instead.

    Coordinates are converted to integers by multiplying by
    ``precision`` and rounding.  The resulting integers are then
    compressed using the XTC mixed-radix bit-packing scheme with
    adaptive run-length encoding of inter-atom deltas.

    Parameters
    ----------
    coords : ndarray, shape (natoms, 3), dtype float32
        Cartesian coordinates for one frame, in nm.  Must be
        C-contiguous.
    precision : float, optional
        Precision factor (default 1000.0).  Coordinates are multiplied
        by this value before rounding to integers.  Higher precision
        yields better accuracy but larger compressed output.

    Returns
    -------
    compressed : bytes
        The compressed coordinate payload, ready to be written into an
        XTC frame.
    minint : tuple of 3 ints
        ``(min_x, min_y, min_z)`` – minimum integer coordinates,
        needed in the frame header.
    maxint : tuple of 3 ints
        ``(max_x, max_y, max_z)`` – maximum integer coordinates,
        needed in the frame header.
    smallidx : int
        Index into the XTC magic-integer table, needed in the frame
        header.

    Raises
    ------
    RuntimeError
        If the C compression routine signals an error.
    MemoryError
        If the internal buffer cannot be allocated.

    Notes
    -----
    * The compression is **lossy**: the maximum per-coordinate error is
      ``0.5 / precision`` nm (0.0005 nm at the default precision).
    * The function allocates an internal buffer of
      ``natoms * 12 + 1024`` bytes, which is always sufficient for the
      compressed output.
    * The returned ``minint``, ``maxint``, and ``smallidx`` must be
      written into the XTC frame header exactly as returned; they are
      required for decompression.

    See Also
    --------
    decompress_coords : The inverse operation.
    write_xtc : High-level writer that calls this automatically.
    """
    cdef int natoms = coords.shape[0]
    cdef int buf_cap = natoms * 3 * 4 + 1024  # generous buffer
    cdef unsigned char *buf = <unsigned char *>malloc(buf_cap)
    if buf == NULL:
        raise MemoryError()

    cdef int out_len = 0
    cdef int out_minint[3]
    cdef int out_maxint[3]
    cdef int out_smallidx = 0

    cdef int rc = xtc_3d_compress(
        <float *>coords.data, natoms, precision,
        buf, buf_cap, &out_len,
        out_minint, out_maxint, &out_smallidx)

    if rc != 0:
        free(buf)
        raise RuntimeError("XTC compression failed")

    cdef bytes result = buf[:out_len]
    free(buf)

    return (result,
            (out_minint[0], out_minint[1], out_minint[2]),
            (out_maxint[0], out_maxint[1], out_maxint[2]),
            out_smallidx)
