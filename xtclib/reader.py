"""
Pure-Python XTC file reader that delegates coordinate decompression
to the compiled Cython/C extension.
"""

import struct
import numpy as np
from xtclib._xtc_core import decompress_coords, compress_coords

# XTC magic number (GROMACS classic format)
XTC_MAGIC = 1995


class XTCFrame:
    """
    A single frame from an XTC trajectory.

    Attributes
    ----------
    step : int
        Integration step number.
    time : float
        Simulation time for this frame, typically in ps.
    box : ndarray, shape (3, 3), dtype float32
        Unit-cell box vectors in nm.  Row *i* is the *i*-th box vector.
        For simulations without periodic boundary conditions this will
        be all zeros.
    coords : ndarray, shape (natoms, 3), dtype float32
        Cartesian atom coordinates in nm.
    natoms : int
        Number of atoms in the frame.

    Notes
    -----
    XTCFrame objects are produced by iterating over an `XTCReader`.
    They are lightweight (``__slots__``-based) and do not copy the
    underlying arrays, so mutating ``frame.coords`` will alter the
    data in-place.

    Examples
    --------
    >>> from xtclib import XTCReader
    >>> with XTCReader("traj.xtc") as reader:
    ...     for frame in reader:
    ...         print(frame)
    XTCFrame(step=0, time=0.0, natoms=140)
    """

    __slots__ = ("step", "time", "box", "coords", "natoms")

    def __init__(self, step, time, box, coords, natoms):
        self.step = step
        self.time = time
        self.box = box
        self.coords = coords
        self.natoms = natoms

    def __repr__(self):
        return (f"XTCFrame(step={self.step}, time={self.time:.1f}, "
                f"natoms={self.natoms})")


class XTCReader:
    """
    Iterator-based reader for XTC trajectory files.

    Reads frames lazily one at a time, keeping memory usage proportional
    to a single frame.  Can be used as a context manager or iterated
    directly.

    Parameters
    ----------
    filename : str or os.PathLike
        Path to the XTC file.

    Notes
    -----
    * The reader supports the GROMACS "classic" XTC format (magic number
      1995).  Files written by any standard GROMACS version are
      compatible.
    * Frames with 9 or fewer atoms are stored as raw floats (no
      compression); the reader handles both cases transparently.
    * The file is opened in binary mode.  If used as a context manager
      the file is closed on exit; if iterated directly without ``with``,
      the file is opened and closed around the iteration automatically.
    * XTC is a *lossy* format: coordinates are stored as integers scaled
      by a precision factor (typically 1000).  This means the effective
      resolution is 0.001 nm (1 pm).

    Examples
    --------
    Context-manager usage (recommended):

    >>> with XTCReader("trajectory.xtc") as reader:
    ...     for frame in reader:
    ...         print(frame.step, frame.coords.shape)
    0 (140, 3)
    1 (140, 3)

    Direct iteration:

    >>> for frame in XTCReader("trajectory.xtc"):
    ...     pass

    Load everything at once:

    >>> data = XTCReader("trajectory.xtc").read_all()
    >>> data["coords"].shape
    (400, 140, 3)
    """

    def __init__(self, filename):
        self.filename = filename
        self._fp = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self._fp = open(self.filename, "rb")
        return self

    def __exit__(self, *exc):
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self):
        own_fp = self._fp is None
        if own_fp:
            self._fp = open(self.filename, "rb")
        try:
            while True:
                frame = self._read_frame()
                if frame is None:
                    break
                yield frame
        finally:
            if own_fp:
                self._fp.close()
                self._fp = None

    # ------------------------------------------------------------------
    # Read all frames at once
    # ------------------------------------------------------------------

    def read_all(self):
        """
        Read every frame and return consolidated NumPy arrays.

        This loads the entire trajectory into memory at once.  For very
        large trajectories where memory is a concern, iterate over the
        reader frame-by-frame instead.

        Returns
        -------
        data : dict
            Dictionary with the following keys:

            ``"coords"`` : ndarray, shape (nframes, natoms, 3), dtype float32
                Atom coordinates in nm.
            ``"time"`` : ndarray, shape (nframes,), dtype float32
                Simulation times (usually ps).
            ``"step"`` : ndarray, shape (nframes,), dtype int32
                Integration step numbers.
            ``"box"`` : ndarray, shape (nframes, 3, 3), dtype float32
                Unit-cell box vectors in nm.
            ``"natoms"`` : int
                Number of atoms per frame.

        Raises
        ------
        ValueError
            If the file contains no frames.
        IOError
            If the file is truncated or corrupt.

        Examples
        --------
        >>> from xtclib import XTCReader
        >>> data = XTCReader("trajectory.xtc").read_all()
        >>> data["coords"].shape
        (400, 140, 3)
        >>> data["time"][[0, -1]]
        array([  0., 399.], dtype=float32)
        """
        frames = list(self)
        if not frames:
            raise ValueError("No frames found in file")
        natoms = frames[0].natoms
        nframes = len(frames)
        coords = np.empty((nframes, natoms, 3), dtype=np.float32)
        times = np.empty(nframes, dtype=np.float32)
        steps = np.empty(nframes, dtype=np.int32)
        boxes = np.empty((nframes, 3, 3), dtype=np.float32)
        for i, f in enumerate(frames):
            coords[i] = f.coords
            times[i] = f.time
            steps[i] = f.step
            boxes[i] = f.box
        return {
            "coords": coords,
            "time": times,
            "step": steps,
            "box": boxes,
            "natoms": natoms,
        }

    # ------------------------------------------------------------------
    # Internal: read one frame
    # ------------------------------------------------------------------

    def _read_frame(self):
        fp = self._fp

        # --- header: magic(4) + natoms(4) + step(4) + time(4) = 16 bytes ---
        hdr = fp.read(16)
        if len(hdr) == 0:
            return None  # EOF
        if len(hdr) < 16:
            raise IOError("Truncated XTC header")

        magic, natoms, step = struct.unpack(">3i", hdr[:12])
        (time_val,) = struct.unpack(">f", hdr[12:16])

        if magic != XTC_MAGIC:
            raise ValueError(f"Bad XTC magic: {magic:#x} (expected {XTC_MAGIC:#x})")

        # --- box: 9 floats = 36 bytes ---
        box_raw = fp.read(36)
        if len(box_raw) < 36:
            raise IOError("Truncated box data")
        box = np.array(struct.unpack(">9f", box_raw), dtype=np.float32).reshape(3, 3)

        # --- For <= 9 atoms the coords are stored uncompressed ---
        if natoms <= 9:
            raw = fp.read(natoms * 3 * 4)
            coords = np.array(
                struct.unpack(f">{natoms * 3}f", raw), dtype=np.float32
            ).reshape(natoms, 3)
            return XTCFrame(step, time_val, box, coords, natoms)

        # --- Compressed frame ------------------------------------------
        # natoms(4) + precision(4)
        (natoms2,) = struct.unpack(">i", fp.read(4))
        (precision,) = struct.unpack(">f", fp.read(4))
        if natoms2 != natoms:
            raise ValueError("Atom count mismatch in XTC frame")

        # minint[3], maxint[3]
        minint = struct.unpack(">3i", fp.read(12))
        maxint = struct.unpack(">3i", fp.read(12))

        # smallidx, compressed size
        (smallidx,) = struct.unpack(">i", fp.read(4))
        (comp_size,) = struct.unpack(">i", fp.read(4))

        # Read compressed payload (XDR opaque: 4-byte aligned)
        comp_data = fp.read(comp_size)
        if len(comp_data) < comp_size:
            raise IOError("Truncated compressed data")
        pad = (4 - comp_size % 4) % 4
        if pad:
            fp.read(pad)

        # Decompress via the C extension
        coords = decompress_coords(
            comp_data, natoms,
            minint[0], minint[1], minint[2],
            maxint[0], maxint[1], maxint[2],
            smallidx, precision,
        )

        return XTCFrame(step, time_val, box, coords, natoms)


def read_xtc(filename):
    """
    Read an entire XTC trajectory file into NumPy arrays.

    This is a convenience wrapper around ``XTCReader.read_all()``.
    All frames are loaded into memory at once.

    Parameters
    ----------
    filename : str or os.PathLike
        Path to the XTC file.

    Returns
    -------
    data : dict
        Dictionary with the following keys:

        ``"coords"`` : ndarray, shape (nframes, natoms, 3), dtype float32
            Atom coordinates in nm.
        ``"time"`` : ndarray, shape (nframes,), dtype float32
            Simulation times (usually ps).
        ``"step"`` : ndarray, shape (nframes,), dtype int32
            Integration step numbers.
        ``"box"`` : ndarray, shape (nframes, 3, 3), dtype float32
            Unit-cell box vectors in nm.
        ``"natoms"`` : int
            Number of atoms per frame.

    Raises
    ------
    ValueError
        If the file contains no frames or has bad magic bytes.
    IOError
        If the file is truncated or corrupt.

    Notes
    -----
    XTC is a lossy format.  Coordinates are stored as integers scaled by
    a precision factor (typically 1000), giving an effective resolution
    of ~0.001 nm.  The values returned here exactly match those produced
    by GROMACS and MDTraj.

    Examples
    --------
    >>> from xtclib import read_xtc
    >>> data = read_xtc("simulation.xtc")
    >>> data["coords"].shape
    (101, 50949, 3)
    >>> data["natoms"]
    50949
    >>> data["time"][:3]
    array([0., 1., 2.], dtype=float32)

    See Also
    --------
    XTCReader : Frame-by-frame iterator for memory-efficient access.
    write_xtc : Write coordinates to a new XTC file.
    """
    reader = XTCReader(filename)
    return reader.read_all()
