#pragma once

#include "field_op.cuh"


static HOST_DEVICE_INLINE void print_storage(const storage *s) {
    for(int i = 0; i < 12; i++) {
        printf("\'0x%08x\', ", s->limbs[i]);
    }
    printf("\n");
}

// ----------------------------------------------------------------------------
// Kernels
// ----------------------------------------------------------------------------

// Adds two affine points and returns the result in affine coordinates
__global__ void add_points_kernel(affine_point *results, const affine_point *points, uint32_t num_points) {

	uint32_t globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t globalStride = blockDim.x * gridDim.x;

	for (uint32_t j = 2 * globalThreadId; j < num_points; j += 2 * globalStride) {

        // Transform into montgomery space
        affine_point a = to_montgomery(&points[j]);

        // Transform into xyzz coordinates
        point_xyzz a_xyzz = to_point_xyzz(&a);

        // Perform mixed addition
	  	point_xyzz res_xyzz = madd_2008_s(&a_xyzz, &points[j + 1]);

        // Transform back to affine coordinates (WARNING: this is expensive)
        affine_point res_affine = to_affine_point(&res_xyzz);

        // Transform back to standard space
        results[j / 2] = from_montgomery(&res_affine);
	}
}

// ----------------------------------------------------------------------------
// Kernel starters
// ----------------------------------------------------------------------------

void add_points(affine_point *ret, const affine_point *points, uint32_t num_points) {


    // init memory
    affine_point *points_d;
    affine_point *results_d;

    cudaMalloc(&points_d, sizeof(affine_point) * num_points);
    cudaMalloc(&results_d, sizeof(affine_point) * num_points / 2);

    cudaMemcpy(points_d, points, sizeof(affine_point) * num_points, cudaMemcpyHostToDevice);

    // Launch the kernel
    add_points_kernel<<<1024, 64>>>(results_d, points_d, num_points);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();


    // Copy result back to host
    cudaMemcpy(ret, results_d, sizeof(affine_point) * num_points / 2, cudaMemcpyDeviceToHost);
}

