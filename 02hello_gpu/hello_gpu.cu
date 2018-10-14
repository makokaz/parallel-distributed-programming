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

__global__ void worker(double * a, long n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    a[i] += i;
  }
}

int main() {
  int n = 100000;
  size_t sz = sizeof(double) * n;
  /* prepare data and copy it to the device */
  double * a_host = (double *)malloc(sz);
  memset(a_host, 0, sz);
  double * a_dev;
  cudaMalloc((void **)&a_dev, sz);
  cudaMemcpy(a_dev, a_host, sz, cudaMemcpyHostToDevice);
  
  /* launch the kernel */
  int block_sz = 256;
  int n_blocks = (n + block_sz - 1) / block_sz;
  worker<<<n_blocks,block_sz>>>(a_dev, n);

  /* get the result back */
  cudaMemcpy(a_host, a_dev, sz, cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; i++) {
    assert(a_host[i] == i);
  }
  printf("OK\n");
  return 0;
}
