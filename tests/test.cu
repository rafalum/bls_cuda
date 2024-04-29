#include <stdlib.h>
#include <gmp.h>
#include <cstdint>
#include "../src/cuda/kernels.cuh"

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


int test_add_points() {

    int num_points = 64 * 1024;
    
    affine_point *points; // Array to hold points
    cudaHostAlloc(&points, num_points * sizeof(affine_point), cudaHostAllocDefault);

    read_points_from_file(points, num_points);

    // Allocate page-locked memory for results
    point_xyzz* results;
    cudaHostAlloc(&results, sizeof(point_xyzz) * num_points / 2, cudaHostAllocDefault);

    add_points(results, points, num_points);

    printf("Results x:\n");
    print_storage(&results[0].x);

    printf("Results y:\n");
    print_storage(&results[0].y);

    return 0;
}


int test_accumulate_points() {

    int num_points = 64 * 1024;
    int num_points_per_thread = 16;
    
    affine_point *points; // Array to hold points
    cudaHostAlloc(&points, num_points * sizeof(affine_point), cudaHostAllocDefault);

    read_points_from_file(points, num_points);

    // Allocate page-locked memory for results
    point_xyzz* results;
    cudaHostAlloc(&results, sizeof(point_xyzz) * num_points / num_points_per_thread, cudaHostAllocDefault);

    accumulate_points(results, points, num_points, num_points_per_thread);

    printf("Results x:\n");
    print_storage(&results[0].x);

    printf("Results y:\n");
    print_storage(&results[0].y);


    // Free allocated memory
    cudaFreeHost(points);
    cudaFreeHost(results);

    return 0;

}

int main() {
    return test_add_points();
}