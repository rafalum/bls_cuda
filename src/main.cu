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
#include <math.h>

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
    int num_results = 32;

    int num_kernel_calls = log2(num_points / num_results);

    bool oldStateRegionA[4] = {false, false, false, false};
    bool oldStateRegionB[4] = {false, false, false, false};

    bool newStateRegionA[4] = {false, false, false, false};
    bool newStateRegionB[4] = {false, false, false, false};

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

    sleep(1);

    bool new_data = false;
    while (running) {

        // Wait for the GPU to finish
        cudaStreamSynchronize(memoryStreamHostToDevice); 
        cudaStreamSynchronize(memoryStreamDeviceToHost);
        cudaDeviceSynchronize();

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
            sem_wait(sem_results_empty);
            memcpy(shm_results, &resA, sizeof(affine_point));
            sem_post(sem_results_full);
        } else if (oldStateRegionB[3]) {
            // new results in resB
            newStateRegionB[3] = false;
            sem_wait(sem_results_empty);
            memcpy(shm_results, &resB, sizeof(affine_point));
            sem_post(sem_results_full);
        }

        // Update state
        for (int i = 0; i < 4; i++) {
            oldStateRegionA[i] = newStateRegionA[i];
            oldStateRegionB[i] = newStateRegionB[i];
        }

    }

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