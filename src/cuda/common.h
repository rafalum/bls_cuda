#define HOST_INLINE __host__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__

#define TLC 12
#define LIMBS_ALIGNMENT(x) ((x) % 4 == 0 ? 16 : ((x) % 2 == 0 ? 8 : 4))

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>


template <unsigned LIMBS_COUNT> struct __align__(LIMBS_ALIGNMENT(LIMBS_COUNT)) ff_storage {
    static constexpr unsigned LC = LIMBS_COUNT;
    uint32_t limbs[LIMBS_COUNT];
};

typedef ff_storage<TLC> storage;

// ----------------------------------------------------------------------------
// Some constants for BLS12-381
// ----------------------------------------------------------------------------

// Modulus for BLS12-381
static constexpr storage MODULUS = {0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe, 0xf6b0f624, 0x6730d2a0, 
                                    0xf38512bf, 0x64774b84, 0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea};
// R^2 = (2^384)^2 mod q
static constexpr storage RR = {0x1c341746, 0xf4df1f34, 0x09d104f1, 0x0a76e6a6, 0x4c95b6d5, 0x8de5476c, 
                               0x939d83c0, 0x67eb88a9, 0xb519952d, 0x9a793e85, 0x92cae3aa, 0x11988fe5};
// INV                                     
static constexpr uint32_t INV = 0xfffcfffd;

static constexpr storage ONE = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};

static constexpr storage ONE_MONTGOMERY = {0x0002fffd, 0x76090000, 0xc40c0002, 0xebf4000b, 0x53c758ba, 0x5f489857, 
                                           0x70525745, 0x77ce5853, 0xa256ec6d, 0x5c071a97, 0xfa80e493, 0x15f65ec3};
