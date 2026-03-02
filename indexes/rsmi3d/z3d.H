// z3d.H – 3-D Morton (Z-order) curve
// Maps grid-cell indices (xi, yi, zi) ∈ [0, 2^num_bits) each to a single
// integer by interleaving bits: bit b of xi → bit 3b, yi → 3b+1, zi → 3b+2.
//
// Usage (internal-level partitioning in RSMI3D):
//   num_bits = ceil(log2(max_partition_num))   // e.g., 4 for bit_num=16
//   Z = compute_Z_value_3d(i, j, k, num_bits)
//   Result ∈ [0, 2^(3*num_bits) - 1]  = [0, bit_num^3 - 1]

#ifndef Z3D_H
#define Z3D_H

long long compute_Z_value_3d(long long xi, long long yi, long long zi, int num_bits);

#endif
