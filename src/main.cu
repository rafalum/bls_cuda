#include "cuda/kernels.cuh"

int main() {
	// Sample random points
    int num_points = 1024 * 64;

    // Allocate page-locked memory for points
	affine_point* points;
    cudaHostAlloc(&points, num_points * sizeof(affine_point), cudaHostAllocDefault);

    // Allocate page-locked memory for results
	affine_point* results;
    cudaHostAlloc(&results, (num_points / 2) * sizeof(affine_point), cudaHostAllocDefault);

    // Two example points
    points[0].x = {0xcfddbdf8, 0x56365dd6, 0xb57b3a3c, 0xcddf676a, 0x72722f04, 0x1d282603, 0xbbeaeb7a, 0xab53c52d, 0x0c423985, 0x40f99e95, 0xf710c136, 0x0cb21ad3};
	points[0].y = {0x22505ca0, 0xf4cfa2a6, 0x7ba4378d, 0x024edad4, 0xdef41b4b, 0x28b1aa29, 0x719d0f96, 0x824ba568, 0x661b1820, 0x27829807, 0xfc392fa7, 0x0942d77a};

    points[1].x = {0xc059f45e, 0xb67bc5e4, 0xab37a3ce, 0x6596f286, 0xcd206c7d, 0x2bc337d9, 0xde4f48d4, 0xf9fa3bc7, 0x4ad369b0, 0x48876aae, 0x17b0ccac, 0x12a7e646};
    points[1].y = {0x6fd429cf, 0x842828d4, 0x63ad6e2f, 0xd42eb999, 0xeffaf51e, 0xb02fb782, 0xf7e82264, 0x258097a7, 0x2bad7acb, 0x62e4b681, 0x51927003, 0x10cb5475};

    for(int i = 2; i < num_points; i+=2) {
        points[i].x = points[0].x;
        points[i].y = points[0].y;

        points[i + 1].x = points[1].x;
        points[i + 1].y = points[1].y;
    }

	add_points(results, points, num_points);

	printf("Results x:\n");
	print_storage(&results[0].x);

	printf("Results y:\n");
	print_storage(&results[0].y);

    // Free allocated memory
    cudaFreeHost(points);
    cudaFreeHost(results);

	return 0;

}