/** 
    @file spmv_csr_cuda.cc
    @brief y = A * x for csr with cuda
*/

/** 
    @brief y = A * x for csr with cuda
    @param (A) a sparse matrix
    @param (vx) a vector
    @param (vy) a vector
    @returns 1 if succeed, 0 if failed
*/

static int spmv_csr_cuda(sparse_t A, vec_t vx, vec_t vy) {

  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:spmv_csr_cuda:\n"
          "write a code that performs SPMV with CSR format in parallel\n"
          "using CUDA.\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);
  
  /* this is a serial code for your reference */
  idx_t M = A.M;
  idx_t * row_start = A.csr.row_start;
  csr_elem_t * elems = A.csr.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) { // conver to kernel + kernel launch
    y[i] = 0.0;
  }
  for (idx_t i = 0; i < M; i++) { // conver to kernel + kernel launch
    idx_t start = row_start[i];
    idx_t end = row_start[i + 1];
    for (idx_t k = start; k < end; k++) {
      csr_elem_t * e = elems + k;
      idx_t j = e->j;
      real  a = e->a;
      y[i] += a * x[j];
    }
  }
  return 1;
}
