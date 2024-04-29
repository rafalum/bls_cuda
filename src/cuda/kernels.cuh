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

// Adds two affine points and returns the result in affine coordinates (this is slow)
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

		// Transform back to affine coordinates (WARNING: this is expensive as we need to calculate the inverses of zz and zzz)
		affine_point res_affine = to_affine_point(&res_xyzz);

		// Transform back to standard space
		results[j / 2] = from_montgomery(&res_affine);
	}
}

// Adds two affine points and returns the result in xyzz coordinates
__global__ void add_points_kernel(point_xyzz *results, const affine_point *points, uint32_t num_points) {

	uint32_t globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t globalStride = blockDim.x * gridDim.x;

	for (uint32_t j = 2 * globalThreadId; j < num_points; j += 2 * globalStride) {

		// Transform into montgomery space
		affine_point a = to_montgomery(&points[j]);

		// Transform into xyzz coordinates
		point_xyzz a_xyzz = to_point_xyzz(&a);

		// Perform mixed addition
		results[j / 2] = madd_2008_s(&a_xyzz, &points[j + 1]);
	}
}

// Adds two affine points and returns the result in xyzz coordinates
__global__ void accumulate_kernel(point_xyzz *results, const affine_point *points, uint32_t num_points) {

	uint32_t globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t globalStride = blockDim.x * gridDim.x;

	uint32_t num_points_to_accumulate = num_points / gridDim.x / blockDim.x;

	for (uint32_t j = 2 * globalThreadId; j < num_points; j += 2 * globalStride * num_points_to_accumulate) {

		// Transform into montgomery space
		affine_point acc = to_montgomery(&points[j]);

		// Transform into xyzz coordinates
		point_xyzz acc_xyzz = to_point_xyzz(&acc);

		for(uint32_t i = 1; i < num_points_to_accumulate; i++) {
			// Accumulate the points
			acc_xyzz = madd_2008_s(&acc_xyzz, &points[j + i]);

		}
        results[j / 2] = acc_xyzz;
	}
}

// ----------------------------------------------------------------------------
// Kernel starters
// ----------------------------------------------------------------------------

/***
 * This function takes in 2n points in affine coordinates and adds pairs of 
 * them together resulting in n points in xyzz coordinates.
 *
 * @param results pointer to the n results
 * @param points pointer to the array of 2n points 
 * @param num_points total number of points in the array
 ***/
void add_points(point_xyzz *results, const affine_point *points, uint32_t num_points) {


    // init memory
    affine_point *points_d;
    point_xyzz *results_d;

    cudaMalloc(&points_d, sizeof(affine_point) * num_points);
    cudaMalloc(&results_d, sizeof(point_xyzz) * num_points / 2);

    cudaMemcpy(points_d, points, sizeof(affine_point) * num_points, cudaMemcpyHostToDevice);

    // Launch the kernel
    add_points_kernel<<<1024, 32>>>(results_d, points_d, num_points);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(results, results_d, sizeof(point_xyzz) * num_points / 2, cudaMemcpyDeviceToHost);
}

/***
 * This function takes in 2n points in affine coordinates and accumulates <num_points_per_thread>
 * points together resulting in (n / <num_points_per_thread>) points in xyzz coordinates.
 *
 * @param results pointer to the n / <num_points_per_thread> results
 * @param points pointer to the array of 2n points 
 * @param num_points total number of points
 * @param num_points_per_thread number of points to be accumulated
 ***/

void accumulate_points(point_xyzz *ret, const affine_point *points, uint32_t num_points, uint32_t num_points_per_thread) {

	// init memory
	affine_point *points_d;
	point_xyzz *results_d;

    cudaMalloc(&points_d, sizeof(affine_point) * num_points);
    cudaMalloc(&results_d, sizeof(point_xyzz) * num_points / num_points_per_thread);

    cudaMemcpy(points_d, points, sizeof(affine_point) * num_points, cudaMemcpyHostToDevice);

    // Launch the kernel (each thread is responsible for <num_points_per_thread> points)
    accumulate_kernel<<<num_points / num_points_per_thread / 32, 32>>>(results_d, points_d, num_points);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(ret, results_d, sizeof(point_xyzz) * num_points / num_points_per_thread, cudaMemcpyDeviceToHost);

}

