/*
 * XTC coordinate (de)compression – standalone C implementation.
 *
 * Based on the algorithm described in the GROMACS XTC format specification.
 * The XTC compression scheme works by:
 *   1. Converting float coords -> integers via precision multiplier
 *   2. Encoding each atom with a "large" encoding (full range)
 *   3. Runs of atoms whose inter-atom deltas are "small" get encoded
 *      with adaptive variable-length mixed-radix bit packing
 *
 * Reference: GROMACS libxdrf.cpp (LGPL-2.1+)
 */

#include "xdr_xtc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

/* ------------------------------------------------------------------ */
/* Magic integers table                                                */
/* ------------------------------------------------------------------ */

/** Index of the first usable entry in magicints[]. Entries before this
 *  are zero and serve as placeholders. */
#define FIRSTIDX 9

/** Theoretical last index (the GROMACS table goes up to 143, but this
 *  implementation only stores entries up to index 72 = 16777216). */
#define LASTIDX  143

/**
 * Magic-integer table used for XTC variable-length encoding.
 *
 * Each entry defines the range of values that can be encoded at that
 * "level".  The encoder/decoder use an index into this table (smallidx)
 * to set the small-delta encoding range.  The values roughly follow a
 * geometric progression with ratio ~2^(1/3), i.e. each step increases
 * the representable range by about 26%.
 *
 * NMAGIC is derived as sizeof(magicints)/sizeof(int) and equals the
 * number of entries actually present in the compiled table.
 */
static const int magicints[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80,
    101, 128, 161, 203, 256, 322, 406, 512, 645, 812,
    1024, 1290, 1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192,
    10321, 13003, 16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570,
    104031, 131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561, 832255,
    1048576, 1321122, 1664510, 2097152, 2642245, 3329021, 4194304, 5284491, 6658042,
    8388607, 10568983, 13316085, 16777216
};

/** Number of entries in the magicints[] table. */
#define NMAGIC ((int)(sizeof(magicints) / sizeof(magicints[0])))

/* ------------------------------------------------------------------ */
/* Bit I/O buffer                                                      */
/* ------------------------------------------------------------------ */

/**
 * Bit-level I/O buffer for reading or writing packed integers.
 *
 * Maintains a byte-index and a sub-byte bit accumulator so that
 * receivebits() / sendbits() can read/write arbitrary numbers of
 * bits across byte boundaries.
 *
 * For reading:  set `data` to the source buffer and leave `wdata` NULL.
 * For writing:  set `wdata` to a zeroed destination buffer and leave
 *               `data` NULL.
 *
 * Fields:
 *   data      – source byte array (read mode), or NULL.
 *   wdata     – destination byte array (write mode), or NULL.
 *   len       – total number of bytes in data/wdata.
 *   index     – current byte position within data/wdata.
 *   lastbits  – number of valid low-order bits in `lastbyte`.
 *   lastbyte  – sub-byte accumulator holding partial bits that have
 *               been consumed from (read) or not yet flushed to
 *               (write) the byte array.
 */
typedef struct {
    const unsigned char *data;   /* for reading  */
    unsigned char       *wdata;  /* for writing  */
    int    len;                  /* total bytes  */
    int    index;                /* byte index   */
    int    lastbits;             /* # valid bits in lastbyte */
    unsigned int lastbyte;       /* accumulator  */
} BitBuf;

/**
 * Initialise a BitBuf for reading from a byte array.
 *
 * @param b     BitBuf to initialise.
 * @param data  Source byte array.
 * @param len   Length of @p data in bytes.
 */
static void bitbuf_init_read(BitBuf *b, const unsigned char *data, int len)
{
    b->data     = data;
    b->wdata    = NULL;
    b->len      = len;
    b->index    = 0;
    b->lastbits = 0;
    b->lastbyte = 0;
}

/**
 * Initialise a BitBuf for writing to a byte array.
 *
 * The destination buffer is zeroed on init.
 *
 * @param b     BitBuf to initialise.
 * @param data  Destination byte array (will be zeroed).
 * @param len   Capacity of @p data in bytes.
 */
static void bitbuf_init_write(BitBuf *b, unsigned char *data, int len)
{
    b->data     = NULL;
    b->wdata    = data;
    b->len      = len;
    b->index    = 0;
    b->lastbits = 0;
    b->lastbyte = 0;
    memset(data, 0, (size_t)len);
}

/* ------------------------------------------------------------------ */
/* receivebits / sendbits – bit-level I/O                              */
/* ------------------------------------------------------------------ */

/**
 * Read an unsigned integer of @p num_of_bits bits from the buffer.
 *
 * Bits are consumed from the byte array in big-endian order: the first
 * bit read is the most-significant bit of the first unconsumed byte.
 * The function handles reads that straddle byte boundaries by using the
 * sub-byte accumulator in the BitBuf.
 *
 * @param b            BitBuf positioned at the next bit to read.
 * @param num_of_bits  Number of bits to read (0..31).
 *
 * @return  The decoded unsigned value.  The upper (32 - num_of_bits)
 *          bits are guaranteed to be zero.
 */
static int receivebits(BitBuf *b, int num_of_bits)
{
    int          num;
    int          lastbits;
    unsigned int lastbyte;
    int          mask = (1 << num_of_bits) - 1;

    lastbits = b->lastbits;
    lastbyte = b->lastbyte;

    num = 0;
    while (num_of_bits >= 8)
    {
        lastbyte = (lastbyte << 8) | b->data[b->index++];
        num |= (int)((lastbyte >> lastbits) << (num_of_bits - 8));
        num_of_bits -= 8;
    }
    if (num_of_bits > 0)
    {
        if (lastbits < num_of_bits)
        {
            lastbits += 8;
            lastbyte  = (lastbyte << 8) | b->data[b->index++];
        }
        lastbits -= num_of_bits;
        num |= (int)((lastbyte >> lastbits) & ((1 << num_of_bits) - 1));
    }
    num &= mask;
    b->lastbits = lastbits;
    b->lastbyte = lastbyte;
    return num;
}

/**
 * Write the low @p num_of_bits bits of @p num into the buffer.
 *
 * Bits are written in big-endian order.  The function handles writes
 * that straddle byte boundaries via the sub-byte accumulator.  After
 * each call, any partially-filled trailing byte is kept in
 * @c b->lastbyte and flushed to @c b->wdata[b->index] so that the
 * buffer is always in a readable state.
 *
 * @param b            BitBuf positioned at the next bit to write.
 * @param num_of_bits  Number of bits to write (0..31).
 * @param num          Value whose low @p num_of_bits bits are written.
 *                     Must be non-negative.
 */
static void sendbits(BitBuf *b, int num_of_bits, int num)
{
    int          lastbits;
    unsigned int lastbyte;

    lastbits = b->lastbits;
    lastbyte = b->lastbyte;

    while (num_of_bits >= 8)
    {
        lastbyte = (lastbyte << 8) | ((unsigned int)(num >> (num_of_bits - 8)) & 0xffu);
        b->wdata[b->index++] = (unsigned char)(lastbyte >> lastbits);
        num_of_bits -= 8;
    }
    if (num_of_bits > 0)
    {
        lastbyte  = (lastbyte << num_of_bits) | (unsigned int)(num & ((1 << num_of_bits) - 1));
        lastbits += num_of_bits;
        if (lastbits >= 8)
        {
            lastbits -= 8;
            b->wdata[b->index++] = (unsigned char)(lastbyte >> lastbits);
        }
    }
    b->lastbits = lastbits;
    b->lastbyte = lastbyte;
    if (lastbits > 0)
    {
        b->wdata[b->index] = (unsigned char)(lastbyte << (8 - lastbits));
    }
}

/* ------------------------------------------------------------------ */
/* sizeofint / sizeofints                                              */
/* ------------------------------------------------------------------ */

/**
 * Return the number of bits needed to represent values in [0, size).
 *
 * Equivalent to ceil(log2(size)) for size > 0.  Returns 0 for size <= 0.
 *
 * @param size  Upper bound of the range (exclusive).
 * @return      Number of bits (0..32).
 */
static int sizeofint(int size)
{
    int num = 1, num_of_bits = 0;
    while (size >= num && num_of_bits < 32)
    {
        num_of_bits++;
        num <<= 1;
    }
    return num_of_bits;
}

/**
 * Compute the total number of bits needed to encode a set of integers
 * using mixed-radix (variable-base) packing.
 *
 * Given @p num_of_ints integers where integer i is in [0, sizes[i]),
 * the set can be packed into a single big number by treating each
 * integer as a digit in a mixed-radix system.  This function
 * calculates the number of bits required to hold that combined value.
 *
 * Internally, the product of sizes is computed in a multi-byte
 * (base-256) representation to avoid overflow.
 *
 * @param num_of_ints  Number of integers to pack (typically 3 for XTC).
 * @param sizes        Array of upper bounds, each element < 2^24 for
 *                     the packed path (checked by caller).
 *
 * @return  Total number of bits required.
 */
static int sizeofints(int num_of_ints, const unsigned int sizes[])
{
    int          i, num;
    int          bytes[32];
    unsigned int num_of_bytes, num_of_bits, bytecnt, tmp;

    num_of_bytes = 1;
    bytes[0]     = 1;
    num_of_bits  = 0;
    for (i = 0; i < num_of_ints; i++)
    {
        tmp = 0;
        for (bytecnt = 0; bytecnt < num_of_bytes; bytecnt++)
        {
            tmp            = (unsigned int)bytes[bytecnt] * sizes[i] + tmp;
            bytes[bytecnt] = (int)(tmp & 0xffu);
            tmp          >>= 8;
        }
        while (tmp != 0)
        {
            bytes[bytecnt++] = (int)(tmp & 0xffu);
            tmp >>= 8;
        }
        num_of_bytes = bytecnt;
    }
    num = 1;
    num_of_bytes--;
    while ((unsigned int)bytes[num_of_bytes] >= (unsigned int)num)
    {
        num_of_bits++;
        num *= 2;
    }
    return (int)(num_of_bits + num_of_bytes * 8);
}

/* ------------------------------------------------------------------ */
/* receiveints / sendints – mixed-radix packed integer I/O              */
/* ------------------------------------------------------------------ */

/**
 * Decode a set of small unsigned integers from a mixed-radix bit pack.
 *
 * This is the inverse of sendints().  It reads @p num_of_bits bits from
 * the buffer, interprets them as a big-endian multi-byte integer, then
 * successively divides by each element of @p sizes (from last to first)
 * to extract individual values.
 *
 * @param buf          BitBuf positioned at the packed data.
 * @param num_of_ints  Number of integers to decode (typically 3).
 * @param num_of_bits  Total number of bits to read (as returned by
 *                     sizeofints() for the same sizes[]).
 * @param sizes        Array of moduli.  Integer i is in [0, sizes[i]).
 * @param nums         (output) Decoded integer values.
 */
static void receiveints(BitBuf *buf, int num_of_ints, int num_of_bits,
                        const unsigned int sizes[], int nums[])
{
    int bytes[32];
    int i, j, num_of_bytes, p, num;

    bytes[0] = bytes[1] = bytes[2] = bytes[3] = 0;
    num_of_bytes = 0;
    while (num_of_bits > 8)
    {
        bytes[num_of_bytes++] = receivebits(buf, 8);
        num_of_bits -= 8;
    }
    if (num_of_bits > 0)
    {
        bytes[num_of_bytes++] = receivebits(buf, num_of_bits);
    }
    for (i = num_of_ints - 1; i > 0; i--)
    {
        num = 0;
        for (j = num_of_bytes - 1; j >= 0; j--)
        {
            num      = (num << 8) | bytes[j];
            p        = (int)((unsigned int)num / sizes[i]);
            bytes[j] = p;
            num      = num - p * (int)sizes[i];
        }
        nums[i] = num;
    }
    nums[0] = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
}

/**
 * Encode a set of small unsigned integers into a mixed-radix bit pack.
 *
 * Combines @p num_of_ints integers into one large number using
 * mixed-radix multiplication (nums[0] + nums[1]*sizes[0] + ...), then
 * writes the result as @p num_of_bits bits to the buffer.
 *
 * The values must satisfy 0 <= nums[i] < sizes[i] for each i.  The
 * maximum supported value of any single sizes[i] is 2^24 - 1
 * (0xFFFFFF) for the packed encoding path; larger ranges are handled
 * by the caller using per-coordinate sendbits() calls instead.
 *
 * @param buf          BitBuf positioned at the write location.
 * @param num_of_ints  Number of integers to encode (typically 3).
 * @param num_of_bits  Total bits to write (from sizeofints()).
 * @param sizes        Array of moduli (upper bounds for each value).
 * @param nums         Array of values to encode.
 */
static void sendints(BitBuf *buf, int num_of_ints, int num_of_bits,
                     const unsigned int sizes[], const unsigned int nums[])
{
    int          i, num_of_bytes, bytecnt;
    unsigned int bytes[32], tmp;

    tmp = nums[0];
    num_of_bytes = 0;
    do
    {
        bytes[num_of_bytes++] = tmp & 0xffu;
        tmp >>= 8;
    } while (tmp != 0);

    for (i = 1; i < num_of_ints; i++)
    {
        tmp = nums[i];
        for (bytecnt = 0; bytecnt < num_of_bytes; bytecnt++)
        {
            tmp            = bytes[bytecnt] * sizes[i] + tmp;
            bytes[bytecnt] = tmp & 0xffu;
            tmp          >>= 8;
        }
        while (tmp != 0)
        {
            bytes[num_of_bytes++] = tmp & 0xffu;
            tmp >>= 8;
        }
        /* num_of_bytes may have grown, but the variable is only updated
           in the inner loop above via bytecnt increment */
        if (num_of_bytes < bytecnt)
            num_of_bytes = bytecnt;
    }

    if (num_of_bits >= num_of_bytes * 8)
    {
        for (i = 0; i < num_of_bytes; i++)
            sendbits(buf, 8, (int)bytes[i]);
        sendbits(buf, num_of_bits - num_of_bytes * 8, 0);
    }
    else
    {
        for (i = 0; i < num_of_bytes - 1; i++)
            sendbits(buf, 8, (int)bytes[i]);
        sendbits(buf, num_of_bits - (num_of_bytes - 1) * 8, (int)bytes[i]);
    }
}

/* ------------------------------------------------------------------ */
/* Macros                                                              */
/* ------------------------------------------------------------------ */

/** Square of x (used for distance² comparisons in run detection). */
#define SQR(x) ((x) * (x))

/** Absolute value of an int. */
static int iabs(int x) { return x < 0 ? -x : x; }
/** Minimum of two ints. */
static int imin(int a, int b) { return a < b ? a : b; }
/** Maximum of two ints. */
static int imax(int a, int b) { return a > b ? a : b; }

/* ------------------------------------------------------------------ */
/* xtc_3d_decompress                                                   */
/* ------------------------------------------------------------------ */

int xtc_3d_decompress(const unsigned char *buf, int buf_len,
                       int natoms, float precision,
                       const int minint[3], const int maxint[3],
                       int smallidx,
                       float *out_coords)
{
    BitBuf bb;
    unsigned int sizeint[3], sizesmall[3], bitsizeint[3];
    int          bitsize;
    int          smaller, smallnum;
    int          i, k, flag, is_smaller, run;
    int          prevcoord[3];
    float        inv_precision;
    float       *lfp;
    int         *ip, *lip, *thiscoord_p;

    if (natoms <= 0 || precision == 0.0f)
        return -1;

    bitbuf_init_read(&bb, buf, buf_len);
    inv_precision = 1.0f / precision;

    /* Allocate integer coordinate buffer */
    ip = (int *)calloc((size_t)natoms * 3, sizeof(int));
    if (!ip)
        return -1;

    sizeint[0] = (unsigned int)(maxint[0] - minint[0] + 1);
    sizeint[1] = (unsigned int)(maxint[1] - minint[1] + 1);
    sizeint[2] = (unsigned int)(maxint[2] - minint[2] + 1);

    /* Check if sizes are too large for packed encoding */
    if ((sizeint[0] | sizeint[1] | sizeint[2]) > 0xffffffu)
    {
        bitsizeint[0] = (unsigned int)sizeofint((int)sizeint[0]);
        bitsizeint[1] = (unsigned int)sizeofint((int)sizeint[1]);
        bitsizeint[2] = (unsigned int)sizeofint((int)sizeint[2]);
        bitsize = 0;  /* flag: use individual encoding */
    }
    else
    {
        bitsizeint[0] = bitsizeint[1] = bitsizeint[2] = 0;
        bitsize = sizeofints(3, sizeint);
    }

    smaller      = magicints[imax(FIRSTIDX, smallidx - 1)] / 2;
    smallnum     = magicints[imin(smallidx, NMAGIC - 1)] / 2;
    sizesmall[0] = sizesmall[1] = sizesmall[2] =
        (unsigned int)magicints[imin(smallidx, NMAGIC - 1)];

    prevcoord[0] = prevcoord[1] = prevcoord[2] = 0;
    lfp = out_coords;
    lip = ip;
    i   = 0;

    while (i < natoms)
    {
        thiscoord_p = lip + (size_t)i * 3;

        if (bitsize == 0)
        {
            thiscoord_p[0] = receivebits(&bb, (int)bitsizeint[0]);
            thiscoord_p[1] = receivebits(&bb, (int)bitsizeint[1]);
            thiscoord_p[2] = receivebits(&bb, (int)bitsizeint[2]);
        }
        else
        {
            receiveints(&bb, 3, bitsize, sizeint, thiscoord_p);
        }

        i++;
        thiscoord_p[0] += minint[0];
        thiscoord_p[1] += minint[1];
        thiscoord_p[2] += minint[2];

        prevcoord[0] = thiscoord_p[0];
        prevcoord[1] = thiscoord_p[1];
        prevcoord[2] = thiscoord_p[2];

        flag       = receivebits(&bb, 1);
        is_smaller = 0;
        if (flag == 1)
        {
            run        = receivebits(&bb, 5);
            is_smaller = run % 3;
            run       -= is_smaller;
            is_smaller--;
        }
        /* when flag == 0, reuse the previous run value (unchanged) */

        if (run > 0)
        {
            thiscoord_p += 3;
            for (k = 0; k < run; k += 3)
            {
                receiveints(&bb, 3, smallidx, sizesmall, thiscoord_p);
                i++;
                thiscoord_p[0] += prevcoord[0] - smallnum;
                thiscoord_p[1] += prevcoord[1] - smallnum;
                thiscoord_p[2] += prevcoord[2] - smallnum;
                if (k == 0)
                {
                    /* Water optimization: swap first two atoms */
                    int tmp;
                    tmp              = thiscoord_p[0];
                    thiscoord_p[0]   = prevcoord[0];
                    prevcoord[0]     = tmp;
                    tmp              = thiscoord_p[1];
                    thiscoord_p[1]   = prevcoord[1];
                    prevcoord[1]     = tmp;
                    tmp              = thiscoord_p[2];
                    thiscoord_p[2]   = prevcoord[2];
                    prevcoord[2]     = tmp;
                    *lfp++ = (float)prevcoord[0] * inv_precision;
                    *lfp++ = (float)prevcoord[1] * inv_precision;
                    *lfp++ = (float)prevcoord[2] * inv_precision;
                }
                else
                {
                    prevcoord[0] = thiscoord_p[0];
                    prevcoord[1] = thiscoord_p[1];
                    prevcoord[2] = thiscoord_p[2];
                }
                *lfp++ = (float)thiscoord_p[0] * inv_precision;
                *lfp++ = (float)thiscoord_p[1] * inv_precision;
                *lfp++ = (float)thiscoord_p[2] * inv_precision;

                thiscoord_p += 3;
            }
        }
        else
        {
            *lfp++ = (float)thiscoord_p[0] * inv_precision;
            *lfp++ = (float)thiscoord_p[1] * inv_precision;
            *lfp++ = (float)thiscoord_p[2] * inv_precision;
        }

        smallidx += is_smaller;
        if (is_smaller < 0)
        {
            smallnum = smaller;
            if (smallidx > FIRSTIDX)
                smaller = magicints[smallidx - 1] / 2;
            else
                smaller = 0;
        }
        else if (is_smaller > 0)
        {
            smaller  = smallnum;
            smallnum = magicints[imin(smallidx, NMAGIC - 1)] / 2;
        }
        sizesmall[0] = sizesmall[1] = sizesmall[2] =
            (unsigned int)magicints[imin(smallidx, NMAGIC - 1)];
    }

    free(ip);
    return 0;
}

/* ------------------------------------------------------------------ */
/* xtc_3d_compress                                                     */
/* ------------------------------------------------------------------ */

int xtc_3d_compress(const float *coords, int natoms, float precision,
                     unsigned char *out_buf, int out_buf_cap, int *out_len,
                     int out_minint[3], int out_maxint[3], int *out_smallidx)
{
    int   *ip;
    int    i, k;
    int    minint[3], maxint[3];
    int    mindiff, diff;
    int    lint1, lint2, lint3, oldlint1, oldlint2, oldlint3;
    unsigned int sizeint[3], sizesmall[3], bitsizeint[3];
    int    bitsize;
    int    smallidx, maxidx, minidx;
    int    smaller, smallnum, larger;
    int    is_small, is_smaller, run, prevrun;
    int   *thiscoord, prevcoord[3];
    unsigned int tmpcoord[30];
    float  lf;
    const float *lfp;
    BitBuf bb;

    if (natoms <= 0 || precision == 0.0f)
        return -1;

    ip = (int *)malloc((size_t)natoms * 3 * sizeof(int));
    if (!ip)
        return -1;

    /* Convert float -> int and find min/max */
    minint[0] = minint[1] = minint[2] = INT_MAX;
    maxint[0] = maxint[1] = maxint[2] = INT_MIN;
    oldlint1 = oldlint2 = oldlint3 = 0;
    mindiff  = INT_MAX;

    lfp = coords;
    for (i = 0; i < natoms; i++)
    {
        if (*lfp >= 0.0f) lf = *lfp * precision + 0.5f;
        else              lf = *lfp * precision - 0.5f;
        lint1 = (int)lf;
        ip[i * 3]     = lint1;
        if (lint1 < minint[0]) minint[0] = lint1;
        if (lint1 > maxint[0]) maxint[0] = lint1;
        lfp++;

        if (*lfp >= 0.0f) lf = *lfp * precision + 0.5f;
        else              lf = *lfp * precision - 0.5f;
        lint2 = (int)lf;
        ip[i * 3 + 1] = lint2;
        if (lint2 < minint[1]) minint[1] = lint2;
        if (lint2 > maxint[1]) maxint[1] = lint2;
        lfp++;

        if (*lfp >= 0.0f) lf = *lfp * precision + 0.5f;
        else              lf = *lfp * precision - 0.5f;
        lint3 = (int)lf;
        ip[i * 3 + 2] = lint3;
        if (lint3 < minint[2]) minint[2] = lint3;
        if (lint3 > maxint[2]) maxint[2] = lint3;
        lfp++;

        diff = iabs(oldlint1 - lint1) + iabs(oldlint2 - lint2) + iabs(oldlint3 - lint3);
        if (diff < mindiff && i > 0)
            mindiff = diff;
        oldlint1 = lint1;
        oldlint2 = lint2;
        oldlint3 = lint3;
    }

    out_minint[0] = minint[0]; out_minint[1] = minint[1]; out_minint[2] = minint[2];
    out_maxint[0] = maxint[0]; out_maxint[1] = maxint[1]; out_maxint[2] = maxint[2];

    sizeint[0] = (unsigned int)(maxint[0] - minint[0] + 1);
    sizeint[1] = (unsigned int)(maxint[1] - minint[1] + 1);
    sizeint[2] = (unsigned int)(maxint[2] - minint[2] + 1);

    if ((sizeint[0] | sizeint[1] | sizeint[2]) > 0xffffffu)
    {
        bitsizeint[0] = (unsigned int)sizeofint((int)sizeint[0]);
        bitsizeint[1] = (unsigned int)sizeofint((int)sizeint[1]);
        bitsizeint[2] = (unsigned int)sizeofint((int)sizeint[2]);
        bitsize = 0;
    }
    else
    {
        bitsizeint[0] = bitsizeint[1] = bitsizeint[2] = 0;
        bitsize = sizeofints(3, sizeint);
    }

    smallidx = FIRSTIDX;
    while (smallidx < NMAGIC - 1 && magicints[smallidx] < mindiff)
        smallidx++;

    *out_smallidx = smallidx;

    bitbuf_init_write(&bb, out_buf, out_buf_cap);

    maxidx   = imin(NMAGIC - 1, smallidx + 8);
    minidx   = maxidx - 8;
    smaller  = magicints[imax(FIRSTIDX, smallidx - 1)] / 2;
    smallnum = magicints[smallidx] / 2;
    sizesmall[0] = sizesmall[1] = sizesmall[2] = (unsigned int)magicints[smallidx];
    larger   = magicints[maxidx] / 2;

    prevcoord[0] = prevcoord[1] = prevcoord[2] = 0;
    prevrun = -1;
    i       = 0;

    while (i < natoms)
    {
        is_small  = 0;
        thiscoord = ip + (size_t)i * 3;

        if (smallidx < maxidx && i >= 1
            && iabs(thiscoord[0] - prevcoord[0]) < larger
            && iabs(thiscoord[1] - prevcoord[1]) < larger
            && iabs(thiscoord[2] - prevcoord[2]) < larger)
        {
            is_smaller = 1;
        }
        else if (smallidx > minidx)
        {
            is_smaller = -1;
        }
        else
        {
            is_smaller = 0;
        }

        if (i + 1 < natoms)
        {
            if (iabs(thiscoord[0] - thiscoord[3]) < smallnum
                && iabs(thiscoord[1] - thiscoord[4]) < smallnum
                && iabs(thiscoord[2] - thiscoord[5]) < smallnum)
            {
                /* Water optimization: swap */
                int tmp;
                tmp = thiscoord[0]; thiscoord[0] = thiscoord[3]; thiscoord[3] = tmp;
                tmp = thiscoord[1]; thiscoord[1] = thiscoord[4]; thiscoord[4] = tmp;
                tmp = thiscoord[2]; thiscoord[2] = thiscoord[5]; thiscoord[5] = tmp;
                is_small = 1;
            }
        }

        tmpcoord[0] = (unsigned int)(thiscoord[0] - minint[0]);
        tmpcoord[1] = (unsigned int)(thiscoord[1] - minint[1]);
        tmpcoord[2] = (unsigned int)(thiscoord[2] - minint[2]);

        if (bitsize == 0)
        {
            sendbits(&bb, (int)bitsizeint[0], (int)tmpcoord[0]);
            sendbits(&bb, (int)bitsizeint[1], (int)tmpcoord[1]);
            sendbits(&bb, (int)bitsizeint[2], (int)tmpcoord[2]);
        }
        else
        {
            sendints(&bb, 3, bitsize, sizeint, tmpcoord);
        }

        prevcoord[0] = thiscoord[0];
        prevcoord[1] = thiscoord[1];
        prevcoord[2] = thiscoord[2];
        thiscoord   += 3;
        i++;

        run = 0;
        if (is_small == 0 && is_smaller == -1)
            is_smaller = 0;

        while (is_small && run < 8 * 3)
        {
            if (is_smaller == -1
                && (SQR(thiscoord[0] - prevcoord[0])
                    + SQR(thiscoord[1] - prevcoord[1])
                    + SQR(thiscoord[2] - prevcoord[2])
                    >= smaller * smaller))
            {
                is_smaller = 0;
            }

            tmpcoord[run++] = (unsigned int)(thiscoord[0] - prevcoord[0] + smallnum);
            tmpcoord[run++] = (unsigned int)(thiscoord[1] - prevcoord[1] + smallnum);
            tmpcoord[run++] = (unsigned int)(thiscoord[2] - prevcoord[2] + smallnum);

            prevcoord[0] = thiscoord[0];
            prevcoord[1] = thiscoord[1];
            prevcoord[2] = thiscoord[2];

            i++;
            thiscoord += 3;
            is_small   = 0;
            if (i < natoms
                && iabs(thiscoord[0] - prevcoord[0]) < smallnum
                && iabs(thiscoord[1] - prevcoord[1]) < smallnum
                && iabs(thiscoord[2] - prevcoord[2]) < smallnum)
            {
                is_small = 1;
            }
        }

        if (run != prevrun || is_smaller != 0)
        {
            prevrun = run;
            sendbits(&bb, 1, 1);
            sendbits(&bb, 5, run + is_smaller + 1);
        }
        else
        {
            sendbits(&bb, 1, 0);
        }

        for (k = 0; k < run; k += 3)
        {
            sendints(&bb, 3, smallidx, sizesmall, &tmpcoord[k]);
        }

        if (is_smaller != 0)
        {
            smallidx += is_smaller;
            if (is_smaller < 0)
            {
                smallnum = smaller;
                smaller  = magicints[imax(FIRSTIDX, smallidx - 1)] / 2;
            }
            else
            {
                smaller  = smallnum;
                smallnum = magicints[imin(smallidx, NMAGIC - 1)] / 2;
            }
            sizesmall[0] = sizesmall[1] = sizesmall[2] =
                (unsigned int)magicints[imin(smallidx, NMAGIC - 1)];
        }
    }

    if (bb.lastbits != 0)
        bb.index++;

    *out_len = bb.index;
    free(ip);
    return 0;
}
