#include <stdlib.h>
#include <gmp.h>
#include <cstdint>
#include <time.h>

#include "host.cu"

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
    int num_intermediate_results_stage1 = 2048;
    int num_intermediate_results_stage2 = 256;
    int num_results = 32;

    bool oldStateRegionA[6] = {false, false, false, false, false, false};
    bool oldStateRegionB[6] = {false, false, false, false, false, false};

    bool newStateRegionA[6] = {false, false, false, false, false, false};
    bool newStateRegionB[6] = {false, false, false, false, false, false};

    // Allocate page-locked memory for points
	affine_point* points;
    cudaHostAlloc(&points, num_points * sizeof(affine_point), cudaHostAllocDefault);

    // Allocate page-locked memory for results
	point_xyzz* reduceA;
    cudaHostAlloc(&reduceA, num_results * sizeof(point_xyzz), cudaHostAllocDefault);
    point_xyzz* reduceB;
    cudaHostAlloc(&reduceB, num_results * sizeof(point_xyzz), cudaHostAllocDefault);


    clock_t start = clock();
    read_points_from_file(points, num_points);
    clock_t end = clock();

    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Function took %f seconds to execute \n", cpu_time_used);

    cudaStream_t memoryStreamHostToDevice, memoryStreamDeviceToHost;
    cudaStream_t kernel1, kernel2, kernel3;
    cudaStreamCreate(&memoryStreamHostToDevice);
    cudaStreamCreate(&memoryStreamDeviceToHost);
    cudaStreamCreate(&kernel1);
    cudaStreamCreate(&kernel2);
    cudaStreamCreate(&kernel3);

    // print_device_properties();

    // init memory
    affine_point *pointsRegionA, *pointsRegionB;
    point_xyzz *intermediateResultsStage1RegionA, *intermediateResultsStage1RegionB;
    point_xyzz *intermediateResultsStage2RegionA, *intermediateResultsStage2RegionB;
    point_xyzz *resultsRegionA, *resultsRegionB;

    affine_point resA;
    affine_point resB;

    cudaMalloc(&pointsRegionA, sizeof(affine_point) * num_points);
    cudaMalloc(&pointsRegionB, sizeof(affine_point) * num_points);
    cudaMalloc(&intermediateResultsStage1RegionA, sizeof(point_xyzz) * num_intermediate_results_stage1);
    cudaMalloc(&intermediateResultsStage1RegionB, sizeof(point_xyzz) * num_intermediate_results_stage1);
    cudaMalloc(&intermediateResultsStage2RegionA, sizeof(point_xyzz) * num_intermediate_results_stage2);
    cudaMalloc(&intermediateResultsStage2RegionB, sizeof(point_xyzz) * num_intermediate_results_stage2);
    cudaMalloc(&resultsRegionA, sizeof(point_xyzz) * num_results);
    cudaMalloc(&resultsRegionB, sizeof(point_xyzz) * num_results);

    printf("Allocated memory\n");

    bool new_data = true;

    for (int i = 0; i < 100; i++) {

        // Wait for the GPU to finish
        cudaStreamSynchronize(memoryStreamHostToDevice); 
        cudaStreamSynchronize(memoryStreamDeviceToHost);
        cudaStreamSynchronize(kernel1);
        cudaStreamSynchronize(kernel2);
        cudaStreamSynchronize(kernel3);

        new_data = true;

        if (i % 7 == 0) {
            new_data = false;
        }

        // Stage 1: Copy points to device
        if (new_data & !oldStateRegionA[0]) {
            cudaMemcpyAsync(pointsRegionA, points, sizeof(affine_point) * num_points, cudaMemcpyHostToDevice, memoryStreamHostToDevice);
            newStateRegionA[0] = true;
        } else if (new_data & !oldStateRegionB[0]) {
            cudaMemcpyAsync(pointsRegionB, points, sizeof(affine_point) * num_points, cudaMemcpyHostToDevice, memoryStreamHostToDevice);
            newStateRegionB[0] = true;
        }

        // Stage 2: Accumulate points
        if (oldStateRegionA[0] & !oldStateRegionA[1]) {
            accumulate_kernel<<<num_intermediate_results_stage1 / 32, 32, 0, kernel1>>>(intermediateResultsStage1RegionA, pointsRegionA, num_points);
            newStateRegionA[0] = false;
            newStateRegionA[1] = true;
        } else if (oldStateRegionB[0] & !oldStateRegionB[1]) {
            accumulate_kernel<<<num_intermediate_results_stage1 / 32, 32, 0, kernel1>>>(intermediateResultsStage1RegionB, pointsRegionB, num_points);
            newStateRegionB[0] = false;
            newStateRegionB[1] = true;
        }

        // Stage 3: Accumulate intermediate results
        if (oldStateRegionA[1] & !oldStateRegionA[2]) {
            accumulate_kernel<<<num_intermediate_results_stage2 / 32, 32, 0, kernel2>>>(intermediateResultsStage2RegionA, intermediateResultsStage1RegionA, num_intermediate_results_stage1);
            newStateRegionA[1] = false;
            newStateRegionA[2] = true;
        } else if (oldStateRegionB[1] & !oldStateRegionB[2]) {
            accumulate_kernel<<<num_intermediate_results_stage2 / 32, 32, 0, kernel2>>>(intermediateResultsStage2RegionB, intermediateResultsStage1RegionB, num_intermediate_results_stage1);
            newStateRegionB[1] = false;
            newStateRegionB[2] = true;
        }

        // Stage 4: Accumulate intermediate results, pt. 2
        if (oldStateRegionA[2] & !oldStateRegionA[3]) {
            accumulate_kernel<<<num_results / 32, 32, 0, kernel3>>>(resultsRegionA, intermediateResultsStage2RegionA, num_intermediate_results_stage2);
            newStateRegionA[2] = false;
            newStateRegionA[3] = true;
        } else if (oldStateRegionB[2] & !oldStateRegionB[3]) {
            accumulate_kernel<<<num_results / 32, 32, 0, kernel3>>>(resultsRegionB, intermediateResultsStage2RegionB, num_intermediate_results_stage2);
            newStateRegionB[2] = false;
            newStateRegionB[3] = true;
        }

        // Stage 5: Copy results to host
        if (oldStateRegionA[3] & !oldStateRegionA[4]) {
            cudaMemcpyAsync(reduceA, resultsRegionA, sizeof(point_xyzz) * num_results, cudaMemcpyDeviceToHost, memoryStreamDeviceToHost);
            newStateRegionA[3] = false;
            newStateRegionA[4] = true;
        } else if (oldStateRegionB[3] & !oldStateRegionB[4]) {
            cudaMemcpyAsync(reduceB, resultsRegionB, sizeof(point_xyzz) * num_results, cudaMemcpyDeviceToHost, memoryStreamDeviceToHost);
            newStateRegionB[3] = false;
            newStateRegionB[4] = true;
        }

        // Stage 6: Reduce results
        if (oldStateRegionA[4] & !oldStateRegionA[5]) {
            resA = host_reduce(reduceA, num_results);
            newStateRegionA[4] = false;
            newStateRegionA[5] = true;
        } else if (oldStateRegionB[4] & !oldStateRegionB[5]) {
            resB = host_reduce(reduceB, num_results);
            newStateRegionB[4] = false;
            newStateRegionB[5] = true;
        }

        if (oldStateRegionA[5]) {
            // new results in resA
            newStateRegionA[5] = false;
        } else if (oldStateRegionB[5]) {
            // new results in resB
            newStateRegionB[5] = false;
        }

        // Update state
        for (int i = 0; i < 6; i++) {
            oldStateRegionA[i] = newStateRegionA[i];
            oldStateRegionB[i] = newStateRegionB[i];
        }

    }

    // Destroy streams
    cudaStreamDestroy(memoryStreamDeviceToHost);
    cudaStreamDestroy(memoryStreamHostToDevice);
    cudaStreamDestroy(kernel1);
    cudaStreamDestroy(kernel2);

    printf("Destroyed streams\n");

    // Free memory 
    cudaFree(pointsRegionA);
    cudaFree(pointsRegionB);
    cudaFree(resultsRegionA);
    cudaFree(resultsRegionB);


    // Free allocated memory
    cudaFreeHost(points);
    cudaFreeHost(resultsRegionA);
    cudaFreeHost(resultsRegionB);

	return 0;

}