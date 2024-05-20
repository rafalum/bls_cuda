# High throughput BLS aggregation

### What is this?

This is a minimalistic implementation of the BLS aggregation algorithm running on a GPU (currently only BLS12-381). 
It is designed to be used in a high-throughput setting, where many aggregation operations are performed in parallel. 

The goal is to aggregate a batch of 64k public keys every 0.5ms in a streaming fashion.

### How does it work?

In order to achieve high throughput, we make use of a 4 stage pipeline with the following stages:
- stage 1: copy `num_points` public keys from the host to the GPU global memory (h_2_d)
- stage 2: launch `log2(num_points / num_results)` addition kernels that repeatedly halve the points to `num_results` results (add)
- stage 3: copy the final results from the GPU global memory to the host (d_2_h)
- stage 4: reduce the final results on the host

To accomodate the pipeline we partition the GPU memory into two regions (A and B), where each region handels alternating batches of public keys.
Here is a visualization of the process:

<div align="center">
    <img src="https://github.com/rafalum/bls_cuda/assets/38735195/0294cbb4-5aa1-4003-a283-14868fa4cd81" width=800 />
</div>
