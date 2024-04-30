#include <stdlib.h>
#include <gmp.h>
#include <cstdint>
#include <time.h>

#include "cuda/kernels.cuh"

void read_points_from_file(affine_point *points, uint32_t num_points) {

    FILE *file;
    char buffer[130];
    mpz_t big_num;

    // Initialize GMP integer
    mpz_init(big_num);

    // Open the file
    file = fopen("bls12-381_points.txt", "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // Read points from the file
    int i = 0;
    while (i < 2 * num_points && fscanf(file, "%s\n", buffer) == 1) {
        mpz_set_str(big_num, buffer, 10); // Read the big number in decimal

        // Extract limbs from the big number
        for (int j = 0; j < TLC; j+=2) {
            mp_limb_t limb = mpz_getlimbn(big_num, j / 2);
            //
            if (i % 2 == 0) {
                points[i / 2].x.limbs[j] = static_cast<uint32_t>(limb & 0xFFFFFFFF);
                points[i / 2].x.limbs[j + 1] = static_cast<uint32_t>((limb >> 32) & 0xFFFFFFFF);

            } else {
                points[i / 2].y.limbs[j] = static_cast<uint32_t>(limb & 0xFFFFFFFF);
                points[i / 2].y.limbs[j + 1] = static_cast<uint32_t>((limb >> 32) & 0xFFFFFFFF);
            }
        }

        i++;
    }

    // Close the file
    fclose(file);
    mpz_clear(big_num);
}


int main() {
	// Sample random points
    int num_points = 1024 * 64;
    int num_points_per_thread = 32;

    // Allocate page-locked memory for points
	affine_point* points;
    cudaHostAlloc(&points, num_points * sizeof(affine_point), cudaHostAllocDefault);

    // Allocate page-locked memory for results
	affine_point* results;
    cudaHostAlloc(&results, (num_points / num_points_per_thread) * sizeof(point_xyzz), cudaHostAllocDefault);


    clock_t start = clock();
    read_points_from_file(points, num_points);
    clock_t end = clock();

    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Function took %f seconds to execute \n", cpu_time_used);

    cudaStream_t memoryStreamHostToDevice, memoryStreamDeviceToHost, runStream;
    cudaStreamCreate(&memoryStreamHostToDevice);
    cudaStreamCreate(&memoryStreamDeviceToHost);
    cudaStreamCreate(&runStream);

    // print_device_properties();

    // init memory
    affine_point *pointsRegionA, *pointsRegionB;
    point_xyzz *resultRegionA, *resultRegionB;

    cudaMalloc(&pointsRegionA, sizeof(affine_point) * num_points);
    cudaMalloc(&pointsRegionB, sizeof(affine_point) * num_points);
    cudaMalloc(&resultRegionA, sizeof(point_xyzz) * (num_points / num_points_per_thread));
    cudaMalloc(&resultRegionB, sizeof(point_xyzz) * (num_points / num_points_per_thread));

    printf("Allocated memory\n");

    for (int i = 0; i < 100; i++) {

        // Wait for the GPU to finish
        cudaStreamSynchronize(memoryStreamHostToDevice); 
        cudaStreamSynchronize(runStream);
        cudaStreamSynchronize(memoryStreamDeviceToHost);

        if(i % 2 == 0) {
            cudaMemcpyAsync(pointsRegionA, points, sizeof(affine_point) * num_points, cudaMemcpyHostToDevice, memoryStreamHostToDevice);
            if(i == 0)
                continue;
            accumulate_kernel<<<64, 32, 0, runStream>>>(resultRegionB, pointsRegionB, num_points);
            cudaMemcpyAsync(results, resultRegionA, sizeof(point_xyzz) * (num_points / num_points_per_thread), cudaMemcpyDeviceToHost, memoryStreamDeviceToHost);
        } else {
            cudaMemcpyAsync(pointsRegionB, points, sizeof(affine_point) * num_points, cudaMemcpyHostToDevice, memoryStreamHostToDevice);
            accumulate_kernel<<<64, 32, 0, runStream>>>(resultRegionA, pointsRegionA, num_points);
            if(i == 1)
                continue;
            cudaMemcpyAsync(results, resultRegionB, sizeof(point_xyzz) * (num_points / num_points_per_thread), cudaMemcpyDeviceToHost, memoryStreamDeviceToHost);
        }


        // Copy result back to host
        //cudaMemcpy(ret, retPtrGPU, sizeof(storage) * (num_points / 2), cudaMemcpyDeviceToHost);

    }

    // Destroy streams
    cudaStreamDestroy(memoryStreamDeviceToHost);
    cudaStreamDestroy(memoryStreamHostToDevice);
    cudaStreamDestroy(runStream);

    printf("Destroyed streams\n");

    // Free memory 
    cudaFree(pointsRegionA);
    cudaFree(pointsRegionB);
    cudaFree(resultRegionA);
    cudaFree(resultRegionB);


    // Free allocated memory
    cudaFreeHost(points);
    cudaFreeHost(results);

	return 0;

}