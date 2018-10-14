/**
   @file cuda_util.h
   @brief small utility functions for cuda
   @author Kenjiro Taura
   @date Oct. 14, 2018
 */

/**
   @brief check if a CUDA API invocation succeeded and show the error msg if any
 */

static void check_cuda_error_(cudaError_t e,
                              const char * msg, const char * file, int line) {
  if (e) {
    fprintf(stderr, "%s:%d:error: %s %s\n",
            file, line, msg, cudaGetErrorString(e));
    exit(1);
  }
}

#define check_cuda_error(e) check_cuda_error_(e, #e, __FILE__, __LINE__)

/**
   @brief check if a kernel invocation succeeded and show the error msg if any
 */

static void check_kernel_error_(const char * msg, const char * file, int line) {
  cudaError_t e = cudaGetLastError();
  if (e) {
    fprintf(stderr, "%s:%d:error: %s %s\n",
            file, line, msg, cudaGetErrorString(e));
    exit(1);
  }
}

/**
   @brief check kernel invocation error
   @details check_kernel_error(kernel-launch-expression)
 */

#define check_kernel_error(exp) do { exp; check_kernel_error_(#exp, __FILE__, __LINE__); } while (0)

/**
   @brief get SM executing the caller
 */
__device__ uint get_smid(void) {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

/**
   @brief get device frequency
 */
static int get_freq() {
  struct cudaDeviceProp prop[1];
  check_cuda_error(cudaGetDeviceProperties(prop, 0));
  return prop->clockRate;
}

/**
   @brief cuda malloc
 */
static void * dev_malloc(size_t sz) {
  void * a = 0;
  cudaError_t e = cudaMalloc(&a, sz);
  if (!a) {
    fprintf(stderr, "error: %s\n", cudaGetErrorString(e));
    exit(1);
  }
  return a;
}

/**
   @brief cuda free
 */
static void dev_free(void * a) {
  cudaFree(a);
}

/**
   @brief copy from dev to host
 */
void to_host(void * dst, void * src, size_t sz) {
  check_cuda_error(cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost));
}

/**
   @brief copy from host to dev
 */
static void to_dev(void * dst, void * src, size_t sz) {
  check_cuda_error(cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice));
}

__device__ inline int get_thread_id_x() {
  return blockDim.x * blockIdx.x + threadIdx.x;
}

__device__ inline int get_thread_id_y() {
  return blockDim.y * blockIdx.y + threadIdx.y;
}

__device__ inline int get_thread_id_z() {
  return blockDim.z * blockIdx.z + threadIdx.z;
}

__device__ inline int get_nthreads_x() {
  return gridDim.x * blockDim.x;
}

__device__ inline int get_nthreads_y() {
  return gridDim.y * blockDim.y;
}

__device__ inline int get_nthreads_z() {
  return gridDim.z * blockDim.z;
}

__device__ inline int get_thread_id() {
  int x = get_thread_id_x();
  int y = get_thread_id_y();
  int z = get_thread_id_z();
  int nx = get_nthreads_x();
  int ny = get_nthreads_y();
  return nx * ny * z + nx * y + x;
}

__device__ inline int get_nthreads() {
  int nx = get_nthreads_x();
  int ny = get_nthreads_y();
  int nz = get_nthreads_z();
  return nx * ny * nz;
}


