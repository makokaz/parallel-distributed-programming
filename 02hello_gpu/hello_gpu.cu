/**
   @file hello_gpu.cu
  */
#include <assert.h>
#include <stdio.h>

/*

  warning : the following code does not check any error returned from
  API calls or kernel launches to keep the textual complexity of the
  program low.  This is a highly discouraged practice.  When you don't
  check errors from kernel launches, your program keeps running and 
  you notice it by wrong results.  

  This code is just to illustrate the concepts you need to master when
  writing CUDA programs.


 */

__global__ void cuda_thread_fun(int n) {
  int i        = blockDim.x * blockIdx.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  if (i < n) {
    printf("hello I am CUDA thread %d out of %d\n", i, nthreads);
  }
}

int main(int argc, char ** argv) {
  int n               = (argc > 1 ? atoi(argv[1]) : 100);
  int thread_block_sz = (argc > 2 ? atoi(argv[2]) : 64);
  int n_thread_blocks = (n + thread_block_sz - 1) / thread_block_sz;

  cuda_thread_fun<<<n_thread_blocks,thread_block_sz>>>(n);
  cudaError_t e = cudaGetLastError();
  if (e) {
    printf("NG: %s\n", cudaGetErrorString(e)); return 1;
  } else {
    printf("OK\n");
  }
  e = cudaThreadSynchronize();
  if (e) {
    printf("NG: %s\n", cudaGetErrorString(e)); return 1;
  } else {
    printf("OK\n");
  }
  return 0;
}
