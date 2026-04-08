"""
XTC trajectory writer – uses the compiled C compression extension.
"""

import struct
import numpy as np
from xtclib._xtc_core import compress_coords

# XTC magic number (GROMACS classic format)
XTC_MAGIC = 1995


def write_xtc(filename, coords, time=None, step=None, box=None, precision=1000.0):
    """
    Write an XTC trajectory file with compressed coordinates.

    Coordinates are converted to integers by multiplying by ``precision``
    and rounding, then compressed using the XTC bit-packing algorithm.
    The output is compatible with GROMACS, MDTraj, MDAnalysis, and all
    other tools that read standard XTC files.

    Parameters
    ----------
    filename : str or os.PathLike
        Output file path.  Any existing file is overwritten.
    coords : array_like, shape (nframes, natoms, 3)
        Atom coordinates in **nanometers**.  Will be cast to float32.
    time : array_like, shape (nframes,), optional
        Simulation time for each frame (usually in ps).  If ``None``,
        defaults to ``[0, 1, 2, ...]``.
    step : array_like, shape (nframes,), optional
        Integration step number for each frame.  If ``None``, defaults
        to ``[0, 1, 2, ...]``.
    box : array_like, shape (nframes, 3, 3) or (3, 3), optional
        Unit-cell box vectors in nm.  A single ``(3, 3)`` matrix is
        broadcast to every frame.  If ``None``, all-zero boxes are
        written (appropriate for non-periodic systems).
    precision : float, optional
        Precision factor for the lossy encoding.  The default 1000.0
        gives ~0.001 nm (1 pm) resolution, matching GROMACS defaults.
        Higher values yield better accuracy but larger files.

    Raises
    ------
    ValueError
        If ``coords`` does not have three dimensions or the last
        dimension is not 3.

    Notes
    -----
    * **Lossy compression**: coordinates are rounded to the nearest
      ``1 / precision`` nm.  With the default ``precision=1000.0`` the
      maximum rounding error is 0.0005 nm.
    * **Water optimization**: the XTC encoder automatically detects
      consecutive atoms with small inter-atom distances (e.g. water
      molecules) and swaps their order internally to improve
      compression.  This swap is reversed on reading, so the output
      coordinates are always in the original order.
    * Frames with 9 or fewer atoms are stored uncompressed as raw
      32-bit floats (XTC specification requirement).
    * The output file is written using the "classic" XTC magic number
      (1995) and 32-bit data sizes.  Systems up to ~300 M atoms are
      supported.

    Examples
    --------
    Write a trajectory and read it back:

    >>> import numpy as np
    >>> from xtclib import write_xtc, read_xtc
    >>> coords = np.random.randn(10, 100, 3).astype(np.float32)
    >>> write_xtc("output.xtc", coords)
    >>> data = read_xtc("output.xtc")
    >>> data["coords"].shape
    (10, 100, 3)

    Round-trip an existing trajectory:

    >>> data = read_xtc("input.xtc")
    >>> write_xtc("copy.xtc", data["coords"], time=data["time"],
    ...           step=data["step"], box=data["box"])

    Use higher precision for sub-picometer accuracy:

    >>> write_xtc("hires.xtc", coords, precision=10000.0)

    See Also
    --------
    read_xtc : Read an XTC file.
    XTCReader : Frame-by-frame reader.
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must have shape (nframes, natoms, 3)")

    nframes, natoms, _ = coords.shape

    if time is None:
        time = np.arange(nframes, dtype=np.float32)
    else:
        time = np.asarray(time, dtype=np.float32)

    if step is None:
        step = np.arange(nframes, dtype=np.int32)
    else:
        step = np.asarray(step, dtype=np.int32)

    if box is None:
        box = np.zeros((nframes, 3, 3), dtype=np.float32)
    else:
        box = np.asarray(box, dtype=np.float32)
        if box.ndim == 2:
            box = np.broadcast_to(box, (nframes, 3, 3)).copy()

    with open(filename, "wb") as fp:
        for i in range(nframes):
            _write_frame(fp, natoms, step[i], time[i],
                         box[i], coords[i], precision)


def _write_frame(fp, natoms, step_val, time_val, box, frame_coords, precision):
    """
    Write a single XTC frame to an open file handle.

    This is an internal helper called by `write_xtc` for each frame.
    It writes the 16-byte header, 36-byte box, and either raw floats
    (for <= 9 atoms) or the compressed coordinate payload.

    Parameters
    ----------
    fp : file object
        Binary file handle open for writing.
    natoms : int
        Number of atoms.
    step_val : int
        Integration step number.
    time_val : float
        Simulation time.
    box : ndarray, shape (3, 3)
        Box vectors for this frame.
    frame_coords : ndarray, shape (natoms, 3), dtype float32
        Coordinates for this frame in nm.
    precision : float
        Precision factor.
    """
    # Header: magic + natoms + step + time
    fp.write(struct.pack(">i", XTC_MAGIC))
    fp.write(struct.pack(">i", natoms))
    fp.write(struct.pack(">i", int(step_val)))
    fp.write(struct.pack(">f", float(time_val)))

    # Box: 3x3 floats
    for r in range(3):
        for c in range(3):
            fp.write(struct.pack(">f", float(box[r, c])))

    # For <= 9 atoms, write uncompressed floats
    if natoms <= 9:
        for a in range(natoms):
            for d in range(3):
                fp.write(struct.pack(">f", float(frame_coords[a, d])))
        return

    # Compressed frame
    frame_coords = np.ascontiguousarray(frame_coords, dtype=np.float32)
    compressed, minint, maxint, smallidx = compress_coords(frame_coords, precision)

    # natoms + precision
    fp.write(struct.pack(">i", natoms))
    fp.write(struct.pack(">f", precision))

    # minint[3], maxint[3]
    fp.write(struct.pack(">3i", *minint))
    fp.write(struct.pack(">3i", *maxint))

    # smallidx + compressed size
    fp.write(struct.pack(">i", smallidx))
    comp_size = len(compressed)
    fp.write(struct.pack(">i", comp_size))

    # Compressed data (4-byte aligned)
    fp.write(compressed)
    pad = (4 - comp_size % 4) % 4
    if pad:
        fp.write(b"\x00" * pad)
