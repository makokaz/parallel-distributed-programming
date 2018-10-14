/**
   @file hello_gpu.cu
  */
#include <stdio.h>

__global__ void worker(double * a, long n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    a[i] += i;
  }
}

int main() {
  int n = 100000;
  size_t sz = sizeof(double) * n;
  double * a_host = (double *)calloc(sz);
  memset(a_host, 0, sz);
  double * a_dev = (double *)cudaMalloc(sz);
  cudaMemcpy(a_dev, a_host, sz, cudaMemcpyHostToDevice);
  
  int block_sz = 256;
  int n_blocks = (n + block_sz - 1) / block_sz;
  worker<<<n_blocks,block_sz>>>(a_dev, n);
  cudaMemcpy(a_host, a_dev, sz, cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; i++) {
    assert(a_host[i] == i);
  }
  return 0;
}
