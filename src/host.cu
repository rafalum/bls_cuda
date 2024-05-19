#include "host_math.cu"


/***
 * This function performs the final reduction of the accumulated points on the host.
 *
 * @param resultRegion memory address where the GPU results are stored
 * @param num_points number of points to be reduced (depends on the number of threads used in the kernel)
 * @return the final reduced point in affine coordinates
 ***/
affine_point host_reduce(point_xyzz *resultRegion, uint32_t num_points) {
	
	// perform final reduction
    point_xyzz acc = resultRegion[0];
    for (int i = 1; i < num_points; i++) {
        acc = host::add(&acc, &resultRegion[i]);
    }

	affine_point res = host::to_affine_point(&acc);

	return host::from_montgomery(&res);
}