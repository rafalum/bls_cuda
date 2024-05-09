#include <stdlib.h>
#include <gmp.h>
#include <cstdint>
#include <time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <unistd.h>
#include <cstring>
#include <signal.h>

#include "host.cu"


volatile sig_atomic_t running = 1;

void sigint_handler(int sig) {
    running = 0; // Set the flag to false to exit the loop
}


int main() {

    // Register signal handler
    struct sigaction sa;
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);

	// Sample random points
    int num_points = 1024 * 64;
    int num_intermediate_results_stage1 = 2048;
    int num_intermediate_results_stage2 = 256;
    int num_results = 32;

    bool oldStateRegionA[6] = {false, false, false, false, false, false};
    bool oldStateRegionB[6] = {false, false, false, false, false, false};

    bool newStateRegionA[6] = {false, false, false, false, false, false};
    bool newStateRegionB[6] = {false, false, false, false, false, false};

    // Initialize shared memory for points and result
    affine_point *shm_points, *shm_results;
    sem_t *sem_points_full, *sem_points_empty;
    sem_t *sem_results_full, *sem_results_empty;
    int fd_points, fd_results;

    // Open the shared memory
    fd_points = shm_open("/bls_shared_points", O_CREAT | O_RDWR, 0666);
    fd_results = shm_open("/bls_shared_results", O_CREAT | O_RDWR, 0666);
    if (fd_points == -1 || fd_results == -1) {
        perror("shm_open");
        exit(EXIT_FAILURE);
    }

    // Size the shared memory
    if (ftruncate(fd_points, num_points * sizeof(affine_point)) == -1 || ftruncate(fd_results, sizeof(affine_point)) == -1) {
        perror("ftruncate");
        exit(EXIT_FAILURE);
    }

    // Map the shared memory
    shm_points = (affine_point *) mmap(NULL, num_points * sizeof(affine_point), PROT_READ, MAP_SHARED, fd_points, 0);
    shm_results = (affine_point *) mmap(NULL, sizeof(affine_point), PROT_READ | PROT_WRITE, MAP_SHARED, fd_results, 0);
    if (shm_points == MAP_FAILED || shm_results == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }

    // Open semaphores
    sem_points_full = sem_open("/sem_points_full", O_CREAT, 0666, 0);
    sem_points_empty = sem_open("/sem_points_empty", O_CREAT, 0666, 1);
    sem_results_full = sem_open("/sem_results_full", O_CREAT, 0666, 0);
    sem_results_empty = sem_open("/sem_results_empty", O_CREAT, 0666, 1);
    if (sem_points_full == SEM_FAILED || sem_points_empty == SEM_FAILED || sem_results_full == SEM_FAILED || sem_results_empty == SEM_FAILED) {
        perror("sem_open");
        exit(EXIT_FAILURE);
    }


    // Allocate page-locked memory for points
	affine_point* points;
    cudaHostAlloc(&points, num_points * sizeof(affine_point), cudaHostAllocDefault);

    // Allocate page-locked memory for results
	point_xyzz* reduceA;
    cudaHostAlloc(&reduceA, num_results * sizeof(point_xyzz), cudaHostAllocDefault);
    point_xyzz* reduceB;
    cudaHostAlloc(&reduceB, num_results * sizeof(point_xyzz), cudaHostAllocDefault);

    // Create the streams for the 6 stages
    cudaStream_t memoryStreamHostToDevice, memoryStreamDeviceToHost;
    cudaStream_t kernel1, kernel2, kernel3;
    cudaStreamCreate(&memoryStreamHostToDevice);
    cudaStreamCreate(&memoryStreamDeviceToHost);
    cudaStreamCreate(&kernel1);
    cudaStreamCreate(&kernel2);
    cudaStreamCreate(&kernel3);

    // Initialize the different memory regions on the device
    affine_point *pointsRegionA, *pointsRegionB;
    point_xyzz *intermediateResultsStage1RegionA, *intermediateResultsStage1RegionB;
    point_xyzz *intermediateResultsStage2RegionA, *intermediateResultsStage2RegionB;
    point_xyzz *resultsRegionA, *resultsRegionB;

    cudaMalloc(&pointsRegionA, sizeof(affine_point) * num_points);
    cudaMalloc(&pointsRegionB, sizeof(affine_point) * num_points);
    cudaMalloc(&intermediateResultsStage1RegionA, sizeof(point_xyzz) * num_intermediate_results_stage1);
    cudaMalloc(&intermediateResultsStage1RegionB, sizeof(point_xyzz) * num_intermediate_results_stage1);
    cudaMalloc(&intermediateResultsStage2RegionA, sizeof(point_xyzz) * num_intermediate_results_stage2);
    cudaMalloc(&intermediateResultsStage2RegionB, sizeof(point_xyzz) * num_intermediate_results_stage2);
    cudaMalloc(&resultsRegionA, sizeof(point_xyzz) * num_results);
    cudaMalloc(&resultsRegionB, sizeof(point_xyzz) * num_results);

    affine_point resA;
    affine_point resB;

    sleep(1);

    bool new_data = false;
    while (running) {

        // Wait for the GPU to finish
        cudaStreamSynchronize(memoryStreamHostToDevice); 
        cudaStreamSynchronize(memoryStreamDeviceToHost);
        cudaStreamSynchronize(kernel1);
        cudaStreamSynchronize(kernel2);
        cudaStreamSynchronize(kernel3);

        if (sem_trywait(sem_points_full) == -1) {
            // No new data
            new_data = false;
        } else {
            // New data
            new_data = true;
            memcpy(points, shm_points, num_points * sizeof(affine_point));
            sem_post(sem_points_empty);
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
            sem_wait(sem_results_empty);
            memcpy(shm_results, &resA, sizeof(affine_point));
            sem_post(sem_results_full);
        } else if (oldStateRegionB[5]) {
            // new results in resB
            newStateRegionB[5] = false;
            sem_wait(sem_results_empty);
            memcpy(shm_results, &resB, sizeof(affine_point));
            sem_post(sem_results_full);
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


    // Clean up shared memory and semaphores
    munmap(shm_points, num_points * sizeof(affine_point));
    munmap(shm_results, sizeof(affine_point));
    close(fd_points);
    close(fd_results);
    sem_close(sem_points_full);
    sem_close(sem_points_empty);
    shm_unlink("/bls_shared_points");
    shm_unlink("/bls_shared_results");
    sem_unlink("/sem_points_full");
    sem_unlink("/sem_points_empty");
    sem_unlink("/sem_results_full");
    sem_unlink("/sem_results_empty");

	return 0;

}