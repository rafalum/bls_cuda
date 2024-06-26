#include <stdlib.h>
#include <gmp.h>
#include <cstdint>
#include <time.h>
#include <math.h>

#include "../src/host.cu"


void read_points_from_file(affine_point *points, uint32_t num_points, const char* filename) {

    FILE *file;
    char buffer[130];
    mpz_t big_num;

    // Initialize GMP integer
    mpz_init(big_num);

    // Open the file
    file = fopen(filename, "r");
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

__global__ void warmupKernel() {
    float x = 1.0;
    for (int i = 0; i < 100; i++) {
        x += i;
    }
}

/***
 * This function measures the performance of reducing 64k points down to 2k points
 * using one accumulation kernel.
 ***/
int bench_accumulation_kernel() {

    cudaFree(0);

    int num_points = 1024 * 64;
    int num_results = 2048;

    // Allocate page-locked memory for points
    affine_point* points;
    cudaHostAlloc(&points, num_points * sizeof(affine_point), cudaHostAllocDefault);

    // Read points from file
    read_points_from_file(points, num_points, "bls12-381_points.txt");

    // Allocate page-locked memory for results
    point_xyzz* results;
    cudaHostAlloc(&results, num_results * sizeof(point_xyzz), cudaHostAllocDefault);

    // Allocate device memory
    affine_point *points_d;
    point_xyzz *results_d;

    cudaMalloc(&points_d, sizeof(affine_point) * num_points);
    cudaMalloc(&results_d, sizeof(point_xyzz) * num_results);

    cudaMemcpy(points_d, points, sizeof(affine_point) * num_points, cudaMemcpyHostToDevice);

    // Launch a warmup kernel
    warmupKernel<<<32, 32>>>();
    cudaDeviceSynchronize(); 

    // Cuda event to measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the kernel (each thread is responsible for <num_points_per_thread> points)
    accumulate_kernel<<<num_results / 32, 32>>>(results_d, points_d, num_points);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("############# Perfomance #############\n");
    printf("-> accumulation kernel\n");
    printf("Reduced %d to %d in: %f ms\n", num_points, num_results, milliseconds);
    printf("######################################\n");

    // Free memory
    cudaFree(points_d);
    cudaFree(results_d);
    cudaFreeHost(points);
    cudaFreeHost(results);

    return 0;
}


/***
 * This function measures the performance of reducing 64k points down to 2k points
 * using a series of subsequent addition kernels.
 ***/
int bench_addition_kernel(){
    
        cudaFree(0);
    
        int num_points = 1024 * 64;
        int num_results = 2048;

        int num_kernel_calls = log2(num_points / num_results);
    
        // Allocate page-locked memory for points
        affine_point* points;
        cudaHostAlloc(&points, num_points * sizeof(affine_point), cudaHostAllocDefault);
    
        // Read points from file
        read_points_from_file(points, num_points, "bls12-381_points.txt");
    
        // Allocate page-locked memory for results
        point_xyzz* results;
        cudaHostAlloc(&results, num_results * sizeof(point_xyzz), cudaHostAllocDefault);
    
        // Allocate device memory
        affine_point *points_d;
        point_xyzz *results_d;

        cudaMalloc(&points_d, sizeof(affine_point) * num_points);
        cudaMalloc(&results_d, sizeof(point_xyzz) * num_results);

        point_xyzz* intermediate_results[num_kernel_calls - 1];
        for (int i = 0; i < num_kernel_calls - 1; i++) {
            cudaMalloc(&intermediate_results[i], sizeof(point_xyzz) * num_points / pow(2, i + 1));
        }
    
        cudaMemcpy(points_d, points, sizeof(affine_point) * num_points, cudaMemcpyHostToDevice);

        // Launch a warmup kernel
        warmupKernel<<<32, 32>>>();
        cudaDeviceSynchronize(); 

        // Cuda event to measure time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    
        // Launch the first kernel
        add_points_kernel<<<num_points / 2 / 32, 32>>>(intermediate_results[0], points_d, num_points);
        #pragma unroll
        for (int i = 1; i < num_kernel_calls - 1; i++) {
            add_points_kernel<<<num_points / pow(2, i + 1) / 32, 32>>>(intermediate_results[i], intermediate_results[i - 1], num_points / pow(2, i));
        }
        add_points_kernel<<<num_results / 32, 32>>>(results_d, intermediate_results[num_kernel_calls - 2], num_points / pow(2, num_kernel_calls - 1));

        // Wait for the GPU to finish
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("############# Perfomance #############\n");
        printf("-> %d subsequent addition kernels\n", num_kernel_calls);
        printf("Reduced %d to %d in: %f ms\n", num_points, num_results, milliseconds);
        printf("######################################\n");

        // Copy result back to host
        cudaMemcpy(results, results_d, sizeof(point_xyzz) * num_results, cudaMemcpyDeviceToHost);

        // Free memory
        cudaFree(points_d);
        cudaFree(results_d);
        cudaFreeHost(points);
        cudaFreeHost(results);

        return 0;
}

/***
 * This function measures the performance of accumulating 1024 batches of 64k points
 * using the pipelined approach.
 ***/
int bench_pipeline() {

    cudaFree(0);

    int num_batches = 1024;

	// Sample random points
    int num_points = 1024 * 64;
    int num_results = 2;

    int num_kernel_calls = log2(num_points / num_results);

    bool oldStateRegionA[4] = {false, false, false, false};
    bool oldStateRegionB[4] = {false, false, false, false};

    bool newStateRegionA[4] = {false, false, false, false};
    bool newStateRegionB[4] = {false, false, false, false};

    // Allocate page-locked memory for points
	affine_point* points;
    cudaHostAlloc(&points, num_points * sizeof(affine_point), cudaHostAllocDefault);

    // Read points from file
    read_points_from_file(points, num_points, "bls12-381_points.txt");

    // Allocate page-locked memory for results
	point_xyzz* reduceA;
    cudaHostAlloc(&reduceA, num_results * sizeof(point_xyzz), cudaHostAllocDefault);
    point_xyzz* reduceB;
    cudaHostAlloc(&reduceB, num_results * sizeof(point_xyzz), cudaHostAllocDefault);

    // Create the streams for the 6 stages
    cudaStream_t memoryStreamHostToDevice, memoryStreamDeviceToHost;
    cudaStream_t kernel;
    cudaStreamCreate(&memoryStreamHostToDevice);
    cudaStreamCreate(&memoryStreamDeviceToHost);
    cudaStreamCreate(&kernel);

    // Initialize the different memory regions on the device
    affine_point *pointsRegionA, *pointsRegionB;
    point_xyzz *resultsRegionA, *resultsRegionB;

    point_xyzz* intermediateResultsRegionA[num_kernel_calls - 1];
    point_xyzz* intermediateResultsRegionB[num_kernel_calls - 1];

    cudaMalloc(&pointsRegionA, sizeof(affine_point) * num_points);
    cudaMalloc(&pointsRegionB, sizeof(affine_point) * num_points);

    cudaMalloc(&resultsRegionA, sizeof(point_xyzz) * num_results);
    cudaMalloc(&resultsRegionB, sizeof(point_xyzz) * num_results);

    for (int i = 0; i < num_kernel_calls - 1; i++) {
        cudaMalloc(&intermediateResultsRegionA[i], sizeof(point_xyzz) * num_points / pow(2, i + 1));
        cudaMalloc(&intermediateResultsRegionB[i], sizeof(point_xyzz) * num_points / pow(2, i + 1));
    }

    affine_point resA;
    affine_point resB;

    // Launch a warmup kernel
    warmupKernel<<<32, 32>>>();
    cudaDeviceSynchronize(); 

    // take in new data every iteration
    bool new_data = true;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < num_batches; i++) {

        // Wait for the GPU to finish
        cudaStreamSynchronize(memoryStreamHostToDevice); 
        cudaStreamSynchronize(memoryStreamDeviceToHost);
        cudaStreamSynchronize(kernel);


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
            add_points_kernel<<<num_points / 2 / 32, 32, 0, kernel>>>(intermediateResultsRegionA[0], pointsRegionA, num_points);
            #pragma unroll
            for (int i = 1; i < num_kernel_calls - 1; i++) {
                add_points_kernel<<<num_points / pow(2, i + 1) / 32, 32, 0, kernel>>>(intermediateResultsRegionA[i], intermediateResultsRegionA[i - 1], num_points / pow(2, i));
            }
            add_points_kernel<<<num_results / 32, 32, 0, kernel>>>(resultsRegionA, intermediateResultsRegionA[num_kernel_calls - 2], num_points / pow(2, num_kernel_calls - 1));
            newStateRegionA[0] = false;
            newStateRegionA[1] = true;
        } else if (oldStateRegionB[0] & !oldStateRegionB[1]) {
            add_points_kernel<<<num_points / 2 / 32, 32, 0, kernel>>>(intermediateResultsRegionB[0], pointsRegionB, num_points);
            #pragma unroll
            for (int i = 1; i < num_kernel_calls - 1; i++) {
                add_points_kernel<<<num_points / pow(2, i + 1) / 32, 32, 0, kernel>>>(intermediateResultsRegionB[i], intermediateResultsRegionB[i - 1], num_points / pow(2, i));
            }
            add_points_kernel<<<num_results / 32, 32, 0, kernel>>>(resultsRegionB, intermediateResultsRegionB[num_kernel_calls - 2], num_points / pow(2, num_kernel_calls - 1));
            newStateRegionB[0] = false;
            newStateRegionB[1] = true;
        }

        // Stage 3: Copy results to host
        if (oldStateRegionA[1] & !oldStateRegionA[2]) {
            cudaMemcpyAsync(reduceA, resultsRegionA, sizeof(point_xyzz) * num_results, cudaMemcpyDeviceToHost, memoryStreamDeviceToHost);
            newStateRegionA[1] = false;
            newStateRegionA[2] = true;
        } else if (oldStateRegionB[1] & !oldStateRegionB[2]) {
            cudaMemcpyAsync(reduceB, resultsRegionB, sizeof(point_xyzz) * num_results, cudaMemcpyDeviceToHost, memoryStreamDeviceToHost);
            newStateRegionB[1] = false;
            newStateRegionB[2] = true;
        }

        // Stage 4: Reduce results
        if (oldStateRegionA[2] & !oldStateRegionA[3]) {
            resA = host_reduce(reduceA, num_results);
            newStateRegionA[2] = false;
            newStateRegionA[3] = true;
        } else if (oldStateRegionB[2] & !oldStateRegionB[3]) {
            resB = host_reduce(reduceB, num_results);
            newStateRegionB[2] = false;
            newStateRegionB[3] = true;
        }

        if (oldStateRegionA[3]) {
            // new results in resA
            newStateRegionA[3] = false;
        } else if (oldStateRegionB[3]) {
            // new results in resB
            newStateRegionB[3] = false;
        }

        // Update state
        for (int i = 0; i < 4; i++) {
            oldStateRegionA[i] = newStateRegionA[i];
            oldStateRegionB[i] = newStateRegionB[i];
        }

    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("############# Perfomance #############\n");
    printf("Ran in: %f ms\n", milliseconds);
    printf("Accumulations/s: %f\n", num_batches / (milliseconds / 1000));
    printf("Point additions/s %f\n", num_batches * num_points / (milliseconds / 1000));
    printf("######################################\n");

    // Destroy streams
    cudaStreamDestroy(memoryStreamDeviceToHost);
    cudaStreamDestroy(memoryStreamHostToDevice);
    cudaStreamDestroy(kernel);

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

int main() {
    bench_accumulation_kernel();
    bench_addition_kernel();
    bench_pipeline();
    return 0;
}