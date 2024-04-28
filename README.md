# High throughput BLS aggregation

### What is this?

This is a minimalistic implementation of the BLS aggregation algorithm running on a GPU (currently only BLS12-381). 
It is designed to be used in a high-throughput setting, where many aggregation operations are performed in parallel. 

The goal is to aggregate a batch of 64k public keys every 0.5ms in a streaming fashion.
