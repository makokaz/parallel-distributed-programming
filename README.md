parallel-distributed-handson
==========================================

Teaching material for Parallel and Distributed Programming Class (developed during the lecture).

Work in the order of directory numbers (00slurm, 01hello, 02hello_gpu, ...)

 * 00slurm : get familiar with the environment and job submission commands
 * 01hello : play with OpenMP
 * 02hello_gpu : play with CUDA
 * 03spmv : an exercise to parallelize sparse matrix vector multiply
 * 04udr : an example showing how to use user-defined reductions in OpenMP

(Incomplete) Change Log
==========================================

 * fixed spmv.cc so it now compiles with nvcc
 * replace nvcc with its absolute path /usr/local/cuda/bin
 