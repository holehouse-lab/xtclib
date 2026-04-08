# xtclib

[![PyPI version](https://img.shields.io/pypi/v/xtclib)](https://pypi.org/project/xtclib/)
[![PyPI downloads](https://img.shields.io/pypi/dm/xtclib)](https://pypi.org/project/xtclib/)
[![Python versions](https://img.shields.io/pypi/pyversions/xtclib)](https://pypi.org/project/xtclib/)
[![License: LGPL v2.1+](https://img.shields.io/badge/License-LGPL_v2.1+-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)

A fast, standalone Python library for reading and writing
[GROMACS XTC](https://manual.gromacs.org/current/reference-manual/file-formats.html#xtc)
trajectory files.  The coordinate compression and decompression are
implemented in C and wrapped with Cython, giving performance close to
the native GROMACS routines while requiring **only NumPy** at runtime.

## Why make this?

This package was written so we have a simple, stand-alone and low-dependency reference implementation of the XTC library for reading/writing XTC files. In many cases, we work with XTC files and do not need the dependency overhead of large packages which themselves often come with large dependency overheads, so having a reference and lightweight implementation felt like a good idea for improved modularity across our ecosystem. The use case here is specifically where reading/writing XTC files is necessary, but we want to avoid installing mdtraj, mdanalysis, gromacs etc. YMMC in terms of how often this is an issue, of course.

This also removes the need for any kind of topology file to read/write an XTC file - we may in the future OPTIONALLY add this in as a .pdb file to provide a simple objective oriented interface but having separate and independent XTC read/write power is a core basal interface we want access to directly.

## Features

- **Read** XTC files frame-by-frame (iterator) or all-at-once into NumPy
  arrays.
- **Write** XTC files with full compression, producing output that is
  bit-identical to GROMACS.
- **Zero runtime dependencies** beyond NumPy — no GROMACS installation,
  no MDTraj, no MDAnalysis required.
- Handles both small systems (intrinsically disordered proteins) and
  large solvated systems (50 k+ atoms) correctly.
- Pure-C compression core with Cython bindings — no external shared
  libraries to manage.

## Installation

### From PyPI (stable release)

```bash
pip install xtclib
```

### From GitHub (latest development version)

```bash
pip install git+https://github.com/holehouse-lab/xtclib.git
```

### From source (for local development)

```bash
git clone https://github.com/holehouse-lab/xtclib.git
cd xtclib
pip install -e .
```

Build requirements (`setuptools`, `Cython`, `numpy`) are declared in
`pyproject.toml` and installed automatically during the build.

## Quick start

### Reading a trajectory

```python
from xtclib import read_xtc

data = read_xtc("trajectory.xtc")

data["coords"]   # ndarray, shape (nframes, natoms, 3), float32, in nm
data["time"]     # ndarray, shape (nframes,), float32
data["step"]     # ndarray, shape (nframes,), int32
data["box"]      # ndarray, shape (nframes, 3, 3), float32
data["natoms"]   # int
```

### Writing a trajectory

```python
from xtclib import write_xtc

write_xtc(
    "output.xtc",
    coords,                # (nframes, natoms, 3) array in nm
    time=times,            # optional, defaults to [0, 1, 2, ...]
    step=steps,            # optional, defaults to [0, 1, 2, ...]
    box=boxes,             # optional, defaults to zeros
    precision=1000.0,      # optional, default 1000.0
)
```

### Frame-by-frame iteration

For large trajectories where loading everything into memory at once
is impractical, use the `XTCReader` iterator:

```python
from xtclib import XTCReader

with XTCReader("large_trajectory.xtc") as reader:
    for frame in reader:
        print(frame.step, frame.time, frame.coords.shape)
        # frame.coords is (natoms, 3) float32
        # frame.box    is (3, 3)      float32
```

### Round-tripping a file

```python
from xtclib import read_xtc, write_xtc

data = read_xtc("input.xtc")
write_xtc("copy.xtc", data["coords"],
          time=data["time"], step=data["step"], box=data["box"])
```

The output file will be byte-identical to the input.

## API reference

### `read_xtc(filename)`

Read an entire XTC file and return a dict of NumPy arrays.

| Key | Type | Description |
|---|---|---|
| `"coords"` | `ndarray (nframes, natoms, 3)` float32 | Atom coordinates in nm |
| `"time"` | `ndarray (nframes,)` float32 | Simulation time (usually ps) |
| `"step"` | `ndarray (nframes,)` int32 | Integration step number |
| `"box"` | `ndarray (nframes, 3, 3)` float32 | Unit-cell box vectors in nm |
| `"natoms"` | `int` | Number of atoms per frame |

### `write_xtc(filename, coords, time=None, step=None, box=None, precision=1000.0)`

Write coordinates to an XTC file with compression.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | `str` | — | Output file path |
| `coords` | array-like `(nframes, natoms, 3)` | — | Coordinates in nm |
| `time` | array-like `(nframes,)` | `[0, 1, 2, ...]` | Frame times |
| `step` | array-like `(nframes,)` | `[0, 1, 2, ...]` | Step numbers |
| `box` | array-like `(nframes, 3, 3)` or `(3, 3)` | zeros | Box vectors |
| `precision` | `float` | `1000.0` | Precision factor |

### `XTCReader(filename)`

Iterator-based reader.  Yields `XTCFrame` objects with attributes
`step`, `time`, `box`, `coords`, and `natoms`.  Can be used as a
context manager.  Also provides a `read_all()` method that returns the
same dict as `read_xtc()`.

## How XTC compression works

The XTC format uses a lossy compression scheme for 3-D coordinates:

1. **Float → integer**: each coordinate is multiplied by a precision
   factor (default 1000) and rounded to the nearest integer.  This
   gives ~0.001 nm (1 pm) resolution.

2. **Full-range encoding**: each atom's integer coordinates are encoded
   relative to the bounding box (`minint`/`maxint`) using a
   mixed-radix bit-packing scheme.

3. **Run-length delta encoding**: consecutive atoms whose inter-atom
   distances are "small" are encoded as differences from the previous
   atom, using a smaller bit range.  The small range adapts up or down
   across the trajectory to track typical bond lengths.

4. **Water optimisation**: within small-delta runs, the first two atoms
   are swapped (O-H → H-O order) to make the deltas even smaller
   for water molecules.  This is reversed transparently on reading.

The result is typically **10–20× smaller** than raw float arrays for
solvated biomolecular systems.

## Precision and lossy encoding

XTC is a **lossy** format.  The maximum per-coordinate rounding error is
`0.5 / precision` nm:

| Precision | Max error (nm) | Max error (Å) |
|---|---|---|
| 1000 (default) | 0.0005 | 0.005 |
| 10000 | 0.00005 | 0.0005 |
| 100 | 0.005 | 0.05 |

When reading a file that was written with a given precision, `read_xtc`
returns the **exact** integer-reconstructed values — there is no
additional loss on read.  A `read → write → read` round-trip with the
same precision produces bit-identical coordinate arrays.

## Compatibility

Files written by `xtclib` are fully compatible with:

- **GROMACS** (`gmx trjconv`, `gmx trajcat`, etc.)
- **MDTraj** (`mdtraj.load_xtc`)
- **MDAnalysis** (`MDAnalysis.coordinates.XTC`)
- Any tool that reads standard XTC (magic number 1995)

This has been validated by round-trip testing: files written by `xtclib`
produce byte-identical coordinate arrays when read back by MDTraj.

## System size limits

`xtclib` uses the classic XTC format (magic number 1995) with 32-bit
frame sizes, which imposes the following limits:

| Quantity | Limit |
|---|---|
| Atoms per frame | ~300 million (limited by 32-bit integer range) |
| Frames per file | Unlimited (sequential frames, no index) |
| Coordinate range | ±2,147,483 nm at precision 1000 |
| Compressed frame size | ~2 GB per frame |

## Project structure

```
xtclib/
├── xtclib/
│   ├── __init__.py        # Package entry point
│   ├── reader.py          # XTCReader, read_xtc()
│   ├── writer.py          # write_xtc()
│   ├── _xtc_core.pyx      # Cython wrapper
│   ├── xdr_xtc.c          # C compression/decompression
│   └── xdr_xtc.h          # C header
├── tests/
│   └── test_xtclib.py     # 21 tests (pytest)
├── example/
│   ├── synuclein_STARLING.xtc   # 140-atom IDP, 400 frames
│   ├── synuclein_STARLING.pdb   # Reference structure
│   └── lysozyme_md.xtc          # 50,949-atom solvated system, 101 frames
├── setup.py
└── pyproject.toml
```

## Running tests

```bash
python -m pytest tests/ -v
```

The test suite validates:

- Reading both example files (shapes, metadata)
- Exact round-trip (read → write → read) for coords, time, step, box
- Byte-identical file sizes after round-trip
- PDB cross-validation (atom count, first-frame coordinates)
- MDTraj compatibility (our written files read back identically)
- Compression of synthetic/random coordinate data

## License

The C compression algorithm is based on the XTC specification as
implemented in GROMACS (LGPL-2.1+). 
