/*
 * XTC coordinate (de)compression – standalone C implementation.
 *
 * Based on the algorithm from GROMACS (libxdrf.cpp) which is LGPL-2.1+.
 * This is a clean-room re-implementation of the public XTC specification
 * for reading and writing compressed 3-D coordinate data.
 */

#ifndef XDR_XTC_H
#define XDR_XTC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Decompress one frame of XTC-compressed 3-D coordinate data.
 *
 * Reverses the XTC compression algorithm: reads a bit-packed payload
 * produced by the GROMACS XTC encoder (or by xtc_3d_compress()) and
 * recovers floating-point Cartesian coordinates.
 *
 * The algorithm proceeds atom-by-atom.  Each atom's coordinates are
 * first read using a "large" full-range encoding, then the decoder
 * checks for a run of subsequent atoms encoded as "small" deltas
 * relative to the previous atom.  The small-delta range adapts up or
 * down across runs to track the typical inter-atom spacing.
 *
 * A water-molecule optimisation is applied: when a run begins, the
 * first two atoms are swapped internally (H-O vs O-H order) to
 * minimise delta magnitudes.  This swap is reversed during decoding.
 *
 * @param buf         Compressed payload bytes.  This is the raw
 *                    coordinate data extracted from the XTC frame,
 *                    *after* the header fields (magic, natoms, step,
 *                    time, box, precision, minint, maxint, smallidx,
 *                    compressed-size) have already been consumed.
 * @param buf_len     Length of @p buf in bytes.
 * @param natoms      Number of atoms in the frame.  Must be > 0.
 * @param precision   Precision factor from the frame header (typically
 *                    1000.0).  Decompressed integer coordinates are
 *                    divided by this value to yield floats in nm.
 *                    Must not be zero.
 * @param minint      Minimum integer coordinate values [x, y, z],
 *                    read from the frame header.  Defines the lower
 *                    bound of the full-range encoding.
 * @param maxint      Maximum integer coordinate values [x, y, z],
 *                    read from the frame header.
 * @param smallidx    Index into the XTC magic-integer table that sets
 *                    the initial small-delta encoding range.  Read
 *                    from the frame header.
 * @param out_coords  Output buffer, caller-allocated, with room for
 *                    at least natoms * 3 floats.  Coordinates are
 *                    written in [x0,y0,z0, x1,y1,z1, ...] order,
 *                    in nanometres.
 *
 * @return  0 on success, -1 on error (bad natoms, zero precision, or
 *          allocation failure).
 */
int xtc_3d_decompress(const unsigned char *buf, int buf_len,
                       int natoms, float precision,
                       const int minint[3], const int maxint[3],
                       int smallidx,
                       float *out_coords);

/**
 * Compress one frame of 3-D coordinate data using the XTC algorithm.
 *
 * Implements the full XTC compression pipeline:
 *  1. Multiply each float coordinate by @p precision and round to the
 *     nearest integer.
 *  2. Compute the bounding box (minint/maxint) of the integer coords.
 *  3. Choose the adaptive small-delta range (smallidx) based on the
 *     minimum inter-atom Manhattan distance.
 *  4. For each atom, write a full-range "large" encoding, then scan
 *     forward to find a run of atoms whose inter-atom deltas fit
 *     within the current small range; encode those as packed deltas.
 *  5. Adapt the small range up or down for the next group.
 *
 * A water-molecule optimisation swaps the first two atoms in each run
 * (O-H -> H-O) to reduce delta magnitudes, matching the GROMACS
 * reference encoder.  The decoder reverses this swap automatically.
 *
 * @param coords       Input coordinates in nm, laid out as
 *                     [x0,y0,z0, x1,y1,z1, ...] (natoms * 3 floats).
 * @param natoms       Number of atoms.  Must be > 0.
 * @param precision    Precision factor (typically 1000.0).  Coordinates
 *                     are multiplied by this before rounding.  Higher
 *                     values yield better accuracy but larger output.
 *                     Must not be zero.
 * @param out_buf      Caller-allocated output buffer for the compressed
 *                     payload.  A safe size is natoms * 12 + 1024
 *                     bytes.
 * @param out_buf_cap  Capacity of @p out_buf in bytes.
 * @param out_len      (output) Actual number of bytes written to
 *                     @p out_buf.
 * @param out_minint   (output) Minimum integer coordinates [x, y, z].
 *                     Must be written into the XTC frame header.
 * @param out_maxint   (output) Maximum integer coordinates [x, y, z].
 *                     Must be written into the XTC frame header.
 * @param out_smallidx (output) Chosen index into the magic-integer
 *                     table.  Must be written into the XTC frame
 *                     header.
 *
 * @return  0 on success, -1 on error (bad natoms, zero precision, or
 *          allocation failure).
 */
int xtc_3d_compress(const float *coords, int natoms, float precision,
                     unsigned char *out_buf, int out_buf_cap, int *out_len,
                     int out_minint[3], int out_maxint[3], int *out_smallidx);

#ifdef __cplusplus
}
#endif

#endif /* XDR_XTC_H */
