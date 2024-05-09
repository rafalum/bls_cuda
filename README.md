# High throughput BLS aggregation

### What is this?

This is a minimalistic implementation of the BLS aggregation algorithm running on a GPU (currently only BLS12-381). 
It is designed to be used in a high-throughput setting, where many aggregation operations are performed in parallel. 

The goal is to aggregate a batch of 64k public keys every 0.5ms in a streaming fashion.

### How does it work?

In order to achieve high throughput, we make use of a 6 stage pipeline with the following stages:
- stage 1: copy `num_points` public keys from the host to the GPU global memory (h_2_d)
- stage 2: launch kernel that accumulates the public keys to `num_intermediate_results_kernel1` intermediate results (acc1)
- stage 3: launch second kernel that accumulates the intermediate results to another `num_intermediate_results_kernel2` intermediate results (acc2)
- stage 4: launch third kernel that accumulates the intermediate results to `num_results` results (acc3)
- stage 5: copy the final results from the GPU global memory to the host (d_2_h)
- stage 6: reduce the final results on the host

The dimensions of the intermediate results can be choosen freely and should be tweaked for optimal performance depending on you GPU. 
Good values to start with are:
- `num_intermediate_results_kernel1` = 2048
- `num_intermediate_results_kernel2` = 256
- `num_results` = 32

To accomodate the pipeline we partition the GPU memory into two regions, where each region handels alternating batches of public keys.
Here is a visualization of the process:

<div align="center">
    <img src="https://github.com/rafalum/bls_cuda/assets/38735195/64e60d72-e487-4c78-8def-8aa2b6220d4e" width=800 />
</div>
