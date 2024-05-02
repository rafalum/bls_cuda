#pragma once

#include <pthread.h> 

#include "host_math.cu"

// argument for the thread worker
typedef struct {
    point_xyzz *points;
    uint32_t num_points;
} thread_arg;

// thread worker
void *host_reduce_thread(void *arg) {
    thread_arg *actual_args = (thread_arg *)arg;
    point_xyzz acc = actual_args->points[0];

    for (int i = 1; i < actual_args->num_points; i++) {
        acc = host::add(&acc, &actual_args->points[i]);
    }

    point_xyzz *result = (point_xyzz *)malloc(sizeof(point_xyzz));
    *result = acc;
    return result;
}

/***
 * This function performs the final reduction of the accumulated points on the host.
 *
 * @param resultRegion memory address where the GPU results are stored
 * @param num_points number of points to be reduced (depends on the number of threads used in the kernel)
 * @return the final reduced point in affine coordinates
 ***/

affine_point host_reduce(point_xyzz *resultRegion, uint32_t num_points) {

	uint32_t num_threads = 32;


	// create threads to perform reduction in parallel
	pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	thread_arg *args = (thread_arg *)malloc(num_threads * sizeof(thread_arg));

	for (uint32_t i = 0; i < num_threads; i++) {
		args[i].points = &resultRegion[i * num_points / num_threads];
		args[i].num_points = num_points / num_threads;
		pthread_create(&threads[i], NULL, host_reduce_thread, (void *)&args[i]);
	}

	// wait for all threads to finish
	point_xyzz *results;
  	point_xyzz *intermediateResults = (point_xyzz *)malloc(num_threads * sizeof(point_xyzz));
	for (uint32_t i = 0; i < num_threads; i++) {
		pthread_join(threads[i], (void **)&results);
		intermediateResults[i] = *results;
		free(results);
	}

	free(threads);
	free(args);
	
	// perform final reduction
    point_xyzz acc = intermediateResults[0];
    for (int i = 1; i < num_threads; i++) {
        acc = host::add(&acc, &intermediateResults[i]);
    }

	affine_point res = host::to_affine_point(&acc);

	return host::from_montgomery(&res);
}