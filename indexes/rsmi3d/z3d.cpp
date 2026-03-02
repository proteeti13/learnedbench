#include "z3d.H"

// 3-D Morton code: interleave bits of xi, yi, zi.
// bit b of xi → result bit 3b
// bit b of yi → result bit 3b+1
// bit b of zi → result bit 3b+2
long long compute_Z_value_3d(long long xi, long long yi, long long zi, int num_bits)
{
    long long result = 0;
    for (int b = 0; b < num_bits; b++)
    {
        result |= ((xi >> b) & 1LL) << (3 * b);
        result |= ((yi >> b) & 1LL) << (3 * b + 1);
        result |= ((zi >> b) & 1LL) << (3 * b + 2);
    }
    return result;
}
