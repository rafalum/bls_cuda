#include <stdlib.h>
#include <gmp.h>
#include <cstdint>

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

int check_results(const affine_point *expected, const affine_point *actual, uint32_t num_points) {

    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < TLC; j++) {
            if (expected[i].x.limbs[j] != actual[i].x.limbs[j]) {
                printf("Error: expected x[%d][%d] = %u, actual x[%d][%d] = %u\n", i, j, expected[i].x.limbs[j], i, j, actual[i].x.limbs[j]);
                return 1;
            }
            if (expected[i].y.limbs[j] != actual[i].y.limbs[j]) {
                printf("Error: expected y[%d][%d] = %u, actual y[%d][%d] = %u\n", i, j, expected[i].y.limbs[j], i, j, actual[i].y.limbs[j]);
                return 1;
            }
        }
    }
    return 0;
}


int test_add_points() {

    int num_points = 64 * 1024;
    
    affine_point *points; // Array to hold points
    cudaHostAlloc(&points, num_points * sizeof(affine_point), cudaHostAllocDefault);

    read_points_from_file(points, num_points, "bls12-381_points.txt");

    // Allocate page-locked memory for results
    affine_point* results;
    cudaHostAlloc(&results, sizeof(affine_point) * num_points / 2, cudaHostAllocDefault);

    add_points(results, points, num_points);

    affine_point *expected = (affine_point *)malloc(sizeof(affine_point) * num_points / 2);
    read_points_from_file(expected, num_points / 2, "test_add_points.txt");

    if (check_results(expected, results, num_points / 2) != 0) {
        return 1;
    }

    printf("Test Add Points: passed!\n");

    // Free allocated memory
    cudaFreeHost(points);
    cudaFreeHost(results);
    free(expected);

    return 0;
}


int test_accumulate_points() {

    int num_points = 64 * 1024;
    int num_points_per_thread = 32;
    
    affine_point *points; // Array to hold points
    cudaHostAlloc(&points, num_points * sizeof(affine_point), cudaHostAllocDefault);

    read_points_from_file(points, num_points, "bls12-381_points.txt");

    // Allocate page-locked memory for results
    point_xyzz* results;
    cudaHostAlloc(&results, sizeof(point_xyzz) * num_points / num_points_per_thread, cudaHostAllocDefault);

    accumulate_points(results, points, num_points, num_points_per_thread);

    affine_point *results_affine = (affine_point *)malloc(sizeof(affine_point) * num_points / num_points_per_thread);
    for (int i = 0; i < num_points / num_points_per_thread; i++) {
        affine_point result_affine = host::to_affine_point(&results[i]);
        results_affine[i] = host::from_montgomery(&result_affine);
    }

    affine_point *expected = (affine_point *)malloc(sizeof(affine_point) * num_points / num_points_per_thread);
    read_points_from_file(expected, num_points / num_points_per_thread, "test_accumulate_points.txt");

    if (check_results(expected, results_affine, num_points / num_points_per_thread) != 0) {
        return 1;
    }

    printf("Test Accumulate Points: passed!\n");

    // Free allocated memory
    cudaFreeHost(points);
    cudaFreeHost(results);
    free(results_affine);
    free(expected);

    return 0;

}

int test_reduce_points() {
    
    int num_points = 64 * 1024;
    int num_points_per_thread = 32;
    
    affine_point *points; // Array to hold points
    cudaHostAlloc(&points, num_points * sizeof(affine_point), cudaHostAllocDefault);

    read_points_from_file(points, num_points, "bls12-381_points.txt");

    // Allocate page-locked memory for results
    point_xyzz* results;
    cudaHostAlloc(&results, sizeof(point_xyzz) * num_points / num_points_per_thread, cudaHostAllocDefault);

    accumulate_points(results, points, num_points, num_points_per_thread);

    affine_point result = host_reduce(results, num_points / num_points_per_thread);

    affine_point *expected = (affine_point *)malloc(sizeof(affine_point));
    read_points_from_file(expected, 1, "test_reduce_points.txt");

    if (check_results(expected, &result, 1) != 0) {
        return 1;
    }

    printf("Test Reduce Points: passed!\n");

    // Free allocated memory
    cudaFreeHost(points);
    cudaFreeHost(results);
    free(expected);

    return 0;
    
    
}

int main() {
    test_add_points();
    test_accumulate_points();
    test_reduce_points();
}