/**
   @file spmv.cc
   @brief sparse matrix vector multiplication
   @author Kenjiro Taura
   @date Oct. 14, 2018
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>

#if __NVCC__
#include "cuda_util.h"
#endif

/** @brief type of matrix index (i,j,...) */
typedef int idx_t;
/** @brief type of a matrix element */
typedef double real;

/** @brief sparse matrix storage format */
typedef enum {
  sparse_format_coo,        /**< coordinate list */
  sparse_format_coo_sorted, /**< sorted coordinate list */
  sparse_format_csr,        /**< compressed sparse row */
  sparse_format_invalid,    /**< invalid */
} sparse_format_t;

/** @brief spmv matrix algorithm */
typedef enum {
  spmv_algo_serial,             /**< serial */
  spmv_algo_parallel,           /**< simple parallel */
  spmv_algo_cuda,               /**< cuda */
  spmv_algo_invalid             /**< invalid */
} spmv_algo_t;

/** @brief an element of coordinate list (i, j, a) */
typedef struct {
  idx_t i;                      /**< row */
  idx_t j;                      /**< column */
  real a;                       /**< element */
} coo_elem_t;

/** @brief an element of compressed sparse row */
typedef struct {
  idx_t j;                      /**< column */
  real a;                       /**< element */
} csr_elem_t;

/** @brief sparse matrix in coodinate list format */
typedef struct {
  coo_elem_t * elems;           /**< elements array */
#ifdef __NVCC__
  coo_elem_t * elems_dev;
#endif
} coo_t;

/** @brief sparse matrix in compressed row format */
typedef struct {
  idx_t * row_start;            /**< elems[row_start[i]] is the first element of row i */
  csr_elem_t * elems;           /**< elements array */
#ifdef __NVCC__
  idx_t * row_start_dev;        /**< device copy of row_start */
  csr_elem_t * elems_dev;       /**< device copy of elems */
#endif
} csr_t;

/** @brief sparse matrix (in any format) */
typedef struct {
  sparse_format_t format;   /**< format */
  idx_t M;                      /**< number of rows */
  idx_t N;                      /**< number of columns */
  idx_t nnz;                    /**< number of non-zeros */
  union {
    coo_t coo;                  /**< coo or sorted coo */
    csr_t csr;                  /**< csr */
  };
} sparse_t;

/** @brief vector */
typedef struct {
  idx_t n;
  real * elems;
#ifdef __NVCC__
  real * elems_dev;
#endif
} vec_t;

/**
   @brief malloc + check
   @param (sz) size to alloc in bytes
   @return pointer to the allocated memory
   @sa xfree
 */

void * xalloc(size_t sz) {
  void * a = malloc(sz);
  if (!a) {
    perror("malloc");
    exit(1);
  }
  return a;
}

/**
   @brief wrap free
   @param (a) a pointer returned by calling xalloc
   @sa xalloc
 */
void xfree(void * a) {
  free(a);
}

/** 
    @brief destroy coo 
*/
void coo_destroy(sparse_t A) {
  xfree(A.coo.elems);
}

/** 
    @brief destroy csr
*/
void csr_destroy(sparse_t A) {
  xfree(A.csr.elems);
}

/** 
    @brief destroy sparse matrix in any format
*/
void sparse_destroy(sparse_t A) {
  switch (A.format) {
  case sparse_format_coo:
  case sparse_format_coo_sorted:
    coo_destroy(A);
    break;
  case sparse_format_csr:
    csr_destroy(A);
    break;
  default:
    fprintf(stderr, "sparse_destroy: invalid format %d\n", A.format);
    break;
  }
}

/**
   @brief destroy vector
 */
void vec_destroy(vec_t x) {
  xfree(x.elems);
}

/** 
    @brief make a random coo matrix
*/
sparse_t mk_coo_random(idx_t M, idx_t N, idx_t nnz,
                       unsigned short rg[3]) {
  coo_elem_t * elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
  for (idx_t k = 0; k < nnz; k++) {
    idx_t i = nrand48(rg) % M;
    idx_t j = nrand48(rg) % N;
    real  a = erand48(rg);
    coo_elem_t * e = elems + k;
    e->i = i;
    e->j = j;
    e->a = a;
  }
  coo_t coo = { elems };
  sparse_t A = { sparse_format_coo, M, N, nnz, { .coo = coo } };
  return A;
}

/** 
    @brief compare two coo elements 
*/
int coo_elem_cmp(const void * a_, const void * b_) {
  coo_elem_t * a = (coo_elem_t *)a_;
  coo_elem_t * b = (coo_elem_t *)b_;
  if (a->i < b->i) return -1;
  if (a->i > b->i) return 1;
  if (a->j < b->j) return -1;
  if (a->j > b->j) return 1;
  return 0;
}

/** 
    @brief
    convert A to coo_sorted format.
    if in_place is true, update elements of A in place.
 */
sparse_t coo_to_coo_sorted(sparse_t A, int in_place) {
  assert(A.format == sparse_format_coo ||
         A.format == sparse_format_coo_sorted);
  idx_t nnz = A.nnz;
  coo_elem_t * B_elems = 0;
  if (in_place) {
    B_elems = A.coo.elems;
  } else {
    B_elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
    memcpy(B_elems, A.coo.elems, sizeof(coo_elem_t) * nnz);
  }
  if (A.format == sparse_format_coo) {
    qsort((void*)B_elems, nnz, sizeof(coo_elem_t), coo_elem_cmp);
  }
  coo_t coo = { B_elems };
  sparse_t B = { sparse_format_coo_sorted, A.M, A.N, A.nnz, { .coo = coo } };
  return B;
}

/** 
    @brief make a random coo matrix, with elements sorted 
    in the dictionary order of (i, j)
*/
sparse_t mk_coo_sorted_random(idx_t M, idx_t N, idx_t nnz,
                              unsigned short rg[3]) {
  sparse_t A = mk_coo_random(M, N, nnz, rg);
  return coo_to_coo_sorted(A, 1);
}

/**
   @brief coo -> csr
   @details if update_A is true, A's elements will become sorted
   in place as a side effect
 */
sparse_t coo_to_csr(sparse_t A, int update_A) {
  assert(A.format == sparse_format_coo ||
         A.format == sparse_format_coo_sorted);
  sparse_t B = coo_to_coo_sorted(A, update_A);
  idx_t M = B.M;
  idx_t N = B.N;
  idx_t nnz = B.nnz;
  idx_t * row_start = (idx_t *)xalloc(sizeof(idx_t) * (M + 1));
  coo_elem_t * B_elems = B.coo.elems;
  csr_elem_t * C_elems = (csr_elem_t *)xalloc(sizeof(csr_elem_t) * nnz);
  for (idx_t i = 0; i < M + 1; i++) {
    row_start[i] = 0;
  }
  for (idx_t k = 0; k < nnz; k++) {
    coo_elem_t * e = B_elems + k;
    row_start[e->i]++;
    C_elems[k].j = e->j;
    C_elems[k].a = e->a;
  }
  idx_t s = 0;
  for (idx_t i = 0; i < M; i++) {
    idx_t t = s + row_start[i];
    row_start[i] = s;
    s = t;
  }
  row_start[M] = s;
  assert(s == nnz);
  csr_t csr = { row_start, C_elems };
  sparse_t C = { sparse_format_csr, M, N, nnz, { .csr = csr } };
  return C;
}

/**
   @brief csr -> coo
 */
sparse_t csr_to_coo(sparse_t A) {
  assert(A.format == sparse_format_csr);
  idx_t M = A.M;
  idx_t N = A.N;
  idx_t nnz = A.nnz;
  idx_t * row_start = A.csr.row_start;
  csr_elem_t * A_elems = A.csr.elems;
  coo_elem_t * B_elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
  for (idx_t i = 0; i < M; i++) {
    idx_t start = row_start[i];
    idx_t end = row_start[i + 1];
    for (idx_t k = start; k < end; k++) {
      csr_elem_t * e = A_elems + k;
      B_elems[k].i = i;
      B_elems[k].j = e->j;
      B_elems[k].a = e->a;
    }
  }
  coo_t coo = { B_elems };
  sparse_t B = { sparse_format_coo_sorted, M, N, nnz, { .coo = coo } };
  return B;
}

/** 
    @brief make a random csr matrix
*/
sparse_t mk_csr_random(idx_t M, idx_t N, idx_t nnz,
                       unsigned short rg[3]) {
  sparse_t A = mk_coo_sorted_random(M, N, nnz, rg);
  return coo_to_csr(A, 1);
}

/** 
    @brief make a random sparse matrix of kind (coo, csr, etc.)
*/
sparse_t mk_sparse_random(sparse_format_t format,
                          idx_t M, idx_t N, idx_t nnz,
                          unsigned short rg[3]) {
  switch (format) {
  case sparse_format_coo:
    return mk_coo_random(M, N, nnz, rg);
  case sparse_format_coo_sorted:
    return mk_coo_sorted_random(M, N, nnz, rg);
  case sparse_format_csr:
    return mk_csr_random(M, N, nnz, rg);
  default:
    fprintf(stderr, "mk_sparse_random: invalid format %d\n", format);
    sparse_t A = { sparse_format_invalid, 0, 0, 0, { } };
    return A;
  }
}

/** 
    @brief transpose a matrix in coordinate list format
*/
sparse_t coo_transpose(sparse_t A, int in_place) {
  assert(A.format == sparse_format_coo
         || A.format == sparse_format_coo_sorted);
  idx_t nnz = A.nnz;
  coo_elem_t * B_elems = 0;
  if (in_place) {
    B_elems = A.coo.elems;
  } else {
    B_elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
    memcpy(B_elems, A.coo.elems, sizeof(coo_elem_t) * nnz);
  }
  for (idx_t k = 0; k < nnz; k++) {
    idx_t i = B_elems[k].i;
    idx_t j = B_elems[k].j;
    B_elems[k].i = j;
    B_elems[k].j = i;
  }
  coo_t coo = { B_elems };
  sparse_t B = { sparse_format_coo, A.N, A.M, nnz, { .coo = coo } };
  return B;
}

/** 
    @brief transpose a matrix in any format
*/
sparse_t sparse_transpose(sparse_t A) {
  switch (A.format) {
  case sparse_format_coo: {
    return coo_transpose(A, 0);
  }
  case sparse_format_coo_sorted: {
    sparse_t B = coo_transpose(A, 0);
    return coo_to_coo_sorted(B, 1);
  }
  case sparse_format_csr: {
    sparse_t B = csr_to_coo(A);
    sparse_t C = coo_transpose(B, 1);
    sparse_t D = coo_to_csr(C, 1);
    sparse_destroy(B);          // and C
    return D;
  }
  default: {
    fprintf(stderr, "mk_sparse_random: invalid format %d\n", A.format);
    sparse_t A = { sparse_format_invalid, 0, 0, 0, { } };
    return A;
  }
  }
}

#if __NVCC__
int coo_to_dev(sparse_t& A) {
  idx_t nnz = A.nnz;
  size_t sz = sizeof(coo_elem_t) * nnz;
  coo_elem_t * elems = A.coo.elems;
  coo_elem_t * elems_dev = (coo_elem_t *)dev_malloc(sz);
  to_dev(elems_dev, elems, sz);
  A.coo.elems_dev = elems_dev;
  return 1;
}

int csr_to_dev(sparse_t& A) {
  idx_t nnz = A.nnz;
  size_t sz = sizeof(csr_elem_t) * nnz;
  csr_elem_t * elems = A.csr.elems;
  csr_elem_t * elems_dev = (csr_elem_t *)dev_malloc(sz);
  to_dev(elems_dev, elems, sz);
  A.csr.elems_dev = elems_dev;
  return 1;
}


/** 
    @brief send contents to device
*/
int sparse_to_dev(sparse_t& A) {
  switch (A.format) {
  case sparse_format_coo: {
    return coo_to_dev(A);
  }
  case sparse_format_coo_sorted: {
    return coo_to_dev(A);
  }
  case sparse_format_csr: {
    return csr_to_dev(A);
  }
  default: {
    fprintf(stderr, "sparse_to_dev: invalid format %d\n", A.format);
    return 0;
  }
  }
}

int vec_to_dev(vec_t& v) {
  idx_t n = v.n;
  size_t sz = sizeof(real) * n;
  real * elems = v.elems;
  real * elems_dev = (real *)dev_malloc(sz);
  to_dev(elems_dev, elems, sz);
  v.elems_dev = elems_dev;
  return 1;
}
#endif


/** 
    @brief y = A * x in serial for coordinate list format
*/
long spmv_coo_serial(sparse_t A, vec_t vx, vec_t vy) {
  idx_t M = A.M;
  idx_t nnz = A.nnz;
  coo_elem_t * elems = A.coo.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
  for (idx_t k = 0; k < nnz; k++) {
    coo_elem_t * e = elems + k;
    idx_t i = e->i;
    idx_t j = e->j;
    real  a = e->a;
    real ax = a * x[j];
    y[i] += ax;
  }
  return 2 * (long)nnz;
}

/** 
    @brief y = A * x in parallel for coordinate list format
*/
long spmv_coo_parallel(sparse_t A, vec_t vx, vec_t vy) {
  idx_t M = A.M;
  idx_t nnz = A.nnz;
  coo_elem_t * elems = A.coo.elems;
  real * x = vx.elems;
  real * y = vy.elems;
#pragma omp parallel for
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
#pragma omp parallel for
  for (idx_t k = 0; k < nnz; k++) {
    coo_elem_t * e = elems + k;
    idx_t i = e->i;
    idx_t j = e->j;
    real  a = e->a;
    real ax = a * x[j];
#pragma omp atomic
    y[i] += ax;
  }
  return 2 * (long)nnz;
}

#if __NVCC__
__global__ void init_const_dev(vec_t v, real c) {
  idx_t i = get_thread_id_x();
  real * x = v.elems_dev;
  if (i < v.n) {
    x[i] = 0.0;
  }
}

__global__ void spmv_coo_dev(sparse_t A, vec_t vx, vec_t vy) {
  idx_t nnz = A.nnz;
  coo_elem_t * elems = A.coo.elems_dev;
  real * x = vx.elems_dev;
  real * y = vy.elems_dev;

  idx_t k = get_thread_id_x();
  if (k < nnz) {
    coo_elem_t * e = elems + k;
    idx_t i = e->i;
    idx_t j = e->j;
    real  a = e->a;
    real ax = a * x[j];
    atomicAdd(&y[i], ax);
  }
}

/** 
    @brief y = A * x with cuda for coordinate list format
*/
long spmv_coo_cuda(sparse_t A, vec_t vx, vec_t vy) {
  idx_t M = A.M;
  idx_t nnz = A.nnz;

  int init_block_sz = 1024;
  int n_init_blocks = (M + init_block_sz - 1) / init_block_sz;
  check_kernel_error((init_const_dev<<<n_init_blocks,init_block_sz>>>(vy, 0.0)));

  int spmv_block_sz = 1024;
  int n_spmv_blocks = (nnz + spmv_block_sz - 1) / spmv_block_sz;
  check_kernel_error((spmv_coo_dev<<<n_spmv_blocks,spmv_block_sz>>>(A, vx, vy)));
  
  return 2 * (long)nnz;
}
#endif                                 

/** 
    @brief y = A * x for coordinate list format
*/
long spmv_coo(spmv_algo_t algo, sparse_t A, vec_t x, vec_t y) {
  switch (algo) {
  case spmv_algo_serial:
    return spmv_coo_serial(A, x, y);
  case spmv_algo_parallel:
    return spmv_coo_parallel(A, x, y);
#if __NVCC__
  case spmv_algo_cuda:
    return spmv_coo_cuda(A, x, y);
#endif
  default:
    fprintf(stderr, "spmv_coo: invalid algorithm %d\n", algo);
    return -1;
  }
}

/** 
    @brief y = A * x in serial for csr format
*/
long spmv_csr_serial(sparse_t A, vec_t vx, vec_t vy) {
  idx_t M = A.M;
  idx_t nnz = A.nnz;
  idx_t * row_start = A.csr.row_start;
  csr_elem_t * elems = A.csr.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
  for (idx_t i = 0; i < M; i++) {
    idx_t start = row_start[i];
    idx_t end = row_start[i + 1];
    for (idx_t k = start; k < end; k++) {
      csr_elem_t * e = elems + k;
      idx_t j = e->j;
      real  a = e->a;
      y[i] += a * x[j];
    }
  }
  return 2 * (long)nnz;
}

/** 
    @brief y = A * x in parallel for csr format
*/
long spmv_csr_parallel(sparse_t A, vec_t vx, vec_t vy) {
  idx_t M = A.M;
  idx_t nnz = A.nnz;
  idx_t * row_start = A.csr.row_start;
  csr_elem_t * elems = A.csr.elems;
  real * x = vx.elems;
  real * y = vy.elems;
#pragma omp parallel for
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
#pragma omp parallel for
  for (idx_t i = 0; i < M; i++) {
    idx_t start = row_start[i];
    idx_t end = row_start[i + 1];
    for (idx_t k = start; k < end; k++) {
      csr_elem_t * e = elems + k;
      idx_t j = e->j;
      real  a = e->a;
      y[i] += a * x[j];
    }
  }
  return 2 * (long)nnz;
}

/** 
    @brief y = A * x for csr format
*/
long spmv_csr(spmv_algo_t algo, sparse_t A, vec_t x, vec_t y) {
  switch (algo) {
  case spmv_algo_serial:
    return spmv_csr_serial(A, x, y);
  case spmv_algo_parallel:
    return spmv_csr_parallel(A, x, y);
  default:
    fprintf(stderr, "spmv_coo: invalid algorithm %d\n", algo);
    return -1;
  }
}

/** 
    @brief y = A * x
*/
long spmv(spmv_algo_t algo, sparse_t A, vec_t x, vec_t y) {
  assert(x.n == A.N);
  assert(y.n == A.M);
  switch (A.format) {
  case sparse_format_coo:
    return spmv_coo(algo, A, x, y);
  case sparse_format_coo_sorted:
    return spmv_coo(algo, A, x, y);
  case sparse_format_csr:
    return spmv_csr(algo, A, x, y);
  default:
    fprintf(stderr, "spmv: invalid format %d\n", A.format);
    return -1;
  }
}

/** 
    @brief square norm of a vector (serial)
*/
real vec_norm2_serial(vec_t v) {
  real s = 0.0;
  real * x = v.elems;
  idx_t n = v.n;
  for (idx_t i = 0; i < n; i++) {
    s += x[i] * x[i];
  }
  return s;
}

/** 
    @brief square norm of a vector (parallel)
*/
real vec_norm2_parallel(vec_t v) {
  real s = 0.0;
  real * x = v.elems;
  idx_t n = v.n;
#pragma omp parallel for reduction(+:s)
  for (idx_t i = 0; i < n; i++) {
    s += x[i] * x[i];
  }
  return s;
}

#if __NVCC__
__global__ void vec_norm2_dev(vec_t v, real * s) {
  idx_t n = v.n;
  idx_t i = get_thread_id_x();
  real * x = v.elems_dev;
  if (i < n) {
    atomicAdd(s, x[i] * x[i]);
  }
}

/** 
    @brief square norm of a vector (parallel)
*/
real vec_norm2_cuda(vec_t v) {
  idx_t n = v.n;
  int vec_norm2_block_sz = 1024;
  int n_vec_norm2_blocks = (n + vec_norm2_block_sz - 1) / vec_norm2_block_sz;
  real * s_dev = (real *)dev_malloc(sizeof(real));
  check_kernel_error((vec_norm2_dev<<<n_vec_norm2_blocks,vec_norm2_block_sz>>>(v, s_dev)));
  real s = 0.0;
  to_host(&s, s_dev, sizeof(real));
  dev_free(s_dev);
  return s;
}
#endif

real vec_norm2(spmv_algo_t algo, vec_t v) {
  switch(algo) {
  case spmv_algo_serial:
    return vec_norm2_serial(v);
  case spmv_algo_parallel:
    return vec_norm2_parallel(v);
#if __NVCC__
  case spmv_algo_cuda:
    return vec_norm2_cuda(v);
#endif
  default:
    fprintf(stderr, "vec_norm2: invalid algo %d\n", algo);
    return -1.0;
  }
}
  
/** 
    @brief scalar x vector (serial)
*/
int scalar_vec_serial(real k, vec_t v) {
  idx_t n = v.n;
  real * x = v.elems;
  for (idx_t i = 0; i < n; i++) {
    x[i] *= k;
  }
  return 1;
}

/** 
    @brief scalar x vector (parallel)
*/
int scalar_vec_parallel(real k, vec_t v) {
  idx_t n = v.n;
  real * x = v.elems;
#pragma omp parallel for
  for (idx_t i = 0; i < n; i++) {
    x[i] *= k;
  }
  return 1;
}

/** 
    @brief scalar x vector (cuda)
*/
#if __NVCC__
__global__ void scalar_vec_dev(real k, vec_t v) {
  idx_t n = v.n;
  real * x = v.elems_dev;
  idx_t i = get_thread_id_x();
  if (i < n) {
    x[i] *= k;
  }
}

int scalar_vec_cuda(real k, vec_t v) {
  idx_t n = v.n;
  int scalar_vec_block_sz = 1024;
  int n_scalar_vec_blocks = (n + scalar_vec_block_sz - 1) / scalar_vec_block_sz;
  check_kernel_error((scalar_vec_dev<<<n_scalar_vec_blocks,scalar_vec_block_sz>>>(k, v)));
  return 1;
}
#endif

/**
   @brief scalar x vector
 */

int scalar_vec(spmv_algo_t algo, real k, vec_t v) {
  switch(algo) {
  case spmv_algo_serial:
    return scalar_vec_serial(k, v);
  case spmv_algo_parallel:
    return scalar_vec_parallel(k, v);
#if __NVCC__
  case spmv_algo_cuda:
    return scalar_vec_cuda(k, v);
#endif
  default:
    fprintf(stderr, "scalar_vec: invalid algo %d\n", algo);
    return 0;
  }
}
  
/** 
    @brief normalize a vector
*/
real vec_normalize(spmv_algo_t algo, vec_t v) {
  real s2 = vec_norm2(algo, v);
  if (s2 < 0.0) return -1.0;
  real s = sqrt(s2);
  if (!scalar_vec(algo, 1/s, v)) return -1.0;
  return s;
}

/** 
    @brief current time in nano second
*/
long cur_time_ns() {
  struct timespec ts[1];
  clock_gettime(CLOCK_REALTIME, ts);
  return ts->tv_sec * 1000000000L + ts->tv_nsec;
}

/** 
    @brief repeat y = A x; x = tA y; many times
*/
real repeat_spmv(spmv_algo_t algo,
                 sparse_t& A, sparse_t& tA,
                 vec_t& x, vec_t& y, idx_t repeat) {
#if __NVCC__
  if (algo == spmv_algo_cuda) {
    sparse_to_dev(A);
    sparse_to_dev(tA);
    vec_to_dev(x);
    vec_to_dev(y);
  }
#endif
  
  printf("repeat_spmv : warm up + error check\n");
  fflush(stdout);
  if (spmv(algo, A, x, y) < 0.0) {              // y = A x
    return -1.0;
  }
  if (vec_norm2(algo, y) < 0.0) {
    return -1.0;
  }
  if (spmv(algo, tA, y, x) < 0.0) {              // y = A x
    return -1.0;
  }
  if (vec_normalize(algo, x) < 0.0) {
    return -1.0;
  }
  printf("repeat_spmv : start\n");
  fflush(stdout);
  real lambda = 0.0;
  long flops = 0;
  long t0 = cur_time_ns();
  for (idx_t r = 0; r < repeat; r++) {
    flops += spmv(algo,  A, x, y);
    lambda = sqrt(vec_norm2(algo, y));
    flops += spmv(algo, tA, y, x);
    vec_normalize(algo, x);
    flops += 2 * y.n + 3 * y.n;
  }
  long t1 = cur_time_ns();
  long dt = t1 - t0;
  printf("%ld flops in %.9e sec (%.9e GFLOPS)\n",
         flops, dt*1.0e-9, flops/(double)dt);
  return lambda;
}
  
/** 
    @brief make a random vector
*/
vec_t mk_vec_random(idx_t n, unsigned short rg[3]) {
  real * x = (real *)xalloc(sizeof(real) * n);
  for (idx_t i = 0; i < n; i++) {
    x[i] = erand48(rg);
  }
  vec_t v = { n, x };
  return v;
}

/** 
    @brief make a unit-length random vector
*/
vec_t mk_vec_unit_random(idx_t n, unsigned short rg[3]) {
  vec_t x = mk_vec_random(n, rg);
  vec_normalize(spmv_algo_serial, x);
  return x;
}

/** 
    @brief make a zero vector
*/
vec_t mk_vec_zero(idx_t n) {
  real * x = (real *)xalloc(sizeof(real) * n);
  for (idx_t i = 0; i < n; i++) {
    x[i] = 0.0;
  }
  vec_t v = { n, x };
  return v;
}

/** 
    @brief command line option
*/
typedef struct {
  idx_t M;
  idx_t N;
  idx_t nnz;
  long repeat;
  char * format_str;
  char * algo_str;
  sparse_format_t format;
  spmv_algo_t algo;
  long seed;
  int error;
  int help;
} cmdline_options_t;

/** 
    @brief command line options
*/
struct option long_options[] = {
  {"M",         required_argument, 0,  'M' },
  {"N",         required_argument, 0,  'N' },
  {"nnz",       required_argument, 0,  'z' },
  {"repeat",    required_argument, 0,  'r' },
  {"format",    required_argument, 0,  'f' },
  {"algo",      required_argument, 0,  'a' },
  {"seed",      required_argument, 0,  's'},
  {"help",      required_argument, 0,  'h'},
  {0,           0,                 0,  0 }
};

/** 
    @brief default values for command line options
*/
cmdline_options_t default_opts() {
  cmdline_options_t opt;
  opt.M = 100000;
  opt.N = 0;                    // default N = M
  opt.nnz = 0;                  // default is (N * M) * 0.01
  opt.repeat = 5;
  opt.format_str = strdup("coo");
  opt.format = sparse_format_invalid;
  opt.algo_str = strdup("serial");
  opt.algo = spmv_algo_invalid;
  opt.seed = 4567890123;
  opt.error = 0;
  opt.help = 0;
  return opt;
}

/** 
    @brief parse a string for matrix format and return an enum value
*/
sparse_format_t parse_sparse_format(char * s) {
  if (strcasecmp(s, "coo") == 0) {
    return sparse_format_coo;
  } else if (strcasecmp(s, "coo_sorted") == 0) {
    return sparse_format_coo_sorted;
  } else if (strcasecmp(s, "csr") == 0) {
    return sparse_format_csr;
  } else {
    fprintf(stderr, "error: invalid sparse format (%s)\n", s);
    fprintf(stderr, "  must be one of { coo, coo_sorted, csr }\n");
    return sparse_format_invalid;
  }
}

/** 
    @brief parse a string for spmv algo and return an enum value
*/
spmv_algo_t parse_spmv_algo(char * s) {
  if (strcasecmp(s, "serial") == 0) {
    return spmv_algo_serial;
  } else if (strcasecmp(s, "parallel") == 0) {
    return spmv_algo_parallel;
  } else if (strcasecmp(s, "cuda") == 0) {
    return spmv_algo_cuda;
  } else {
    fprintf(stderr, "error: invalid spmv algo (%s)\n", s);
    fprintf(stderr, "  must be one of { serial, parallel }\n");
    return spmv_algo_invalid;
  }
}

/** 
    @brief release memory for cmdline_options
*/
void cmdline_options_destroy(cmdline_options_t opt) {
  xfree(opt.format_str);
  xfree(opt.algo_str);
}

/**
   @brief usage
  */
void usage(const char * prog) {
  cmdline_options_t o = default_opts();
  fprintf(stderr,
          "usage:\n"
          "\n"
          "%s [options ...]\n"
          "\n"
          "options:\n"
          "  --help        show this help\n"
          "  --M N         set the number of rows to N [%ld]\n"
          "  --N N         set the number of colums to N [%ld]\n"
          "  --nnz N       set the number of non-zero elements to N [%ld]\n"
          "  --repeat N    repeat N times [%ld]\n"
          "  --format F    set sparse matrix format to F [%s]\n"
          "  --algo A      set algorithm to A [%s]\n"
          "  --seed S      set random seed to S [%ld]\n"
          ,
          prog,
          (long)o.M, (long)o.N,
          (long)o.nnz, o.repeat, o.format_str, o.algo_str, o.seed);
}

/** 
    @brief parse command line args
*/
cmdline_options_t parse_args(int argc, char ** argv) {
  char * prog = argv[0];
  cmdline_options_t opt = default_opts();
  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "M:N:z:r:f:a:s:h",
                        long_options, &option_index);
    if (c == -1) break;
    switch (c) {
    case 0:
      printf("option %s", long_options[option_index].name);
      if (optarg)
        printf(" with arg %s", optarg);
      printf("\n");
      break;
    case 'M':
      opt.M = atol(optarg);
      break;
    case 'N':
      opt.N = atol(optarg);
      break;
    case 'z':
      opt.nnz = atol(optarg);
      break;
    case 'r':
      opt.repeat = atol(optarg);
      break;
    case 'f':
      opt.format_str = strdup(optarg);
      break;
    case 'a':
      opt.algo_str = strdup(optarg);
      break;
    case 's':
      opt.seed = atol(optarg);
      break;
    case 'h':
      opt.help = 1;
      break;
    default: /* '?' */
      usage(prog);
      opt.error = 1;
      return opt;
    }
  }
  opt.format = parse_sparse_format(opt.format_str);
  if (opt.format == sparse_format_invalid) {
    opt.error = 1;
    return opt;
  }
  opt.algo = parse_spmv_algo(opt.algo_str);
  if (opt.algo == spmv_algo_invalid) {
    opt.error = 1;
    return opt;
  }
  return opt;
}

/** 
    @brief main
*/
int main(int argc, char ** argv) {
  cmdline_options_t opt = parse_args(argc, argv);
  if (opt.help || opt.error) {
    usage(argv[0]);
    exit(opt.error);
  }
  idx_t M        = opt.M;
  idx_t N        = (opt.N ? opt.N : M);
  idx_t nnz      = (opt.nnz ? opt.nnz : ((long)M * (long)N + 99L) / 100L);
  long repeat    = opt.repeat;
  unsigned short rg[3] = {
    (unsigned short)((opt.seed >> 32) & ((1 << 16) - 1)),
    (unsigned short)((opt.seed >> 16) & ((1 << 16) - 1)),
    (unsigned short)((opt.seed >> 0 ) & ((1 << 16) - 1)),
  };
  long flops = 2 * 2 * nnz * repeat;
  printf("A : %ld x %ld, %ld non-zeros %ld bytes for non-zeros\n",
         (long)M, (long)N, (long)nnz, nnz * sizeof(real));
  printf("repeat : %ld times\n", repeat);
  printf("format : %s\n", opt.format_str);
  printf("algo : %s\n", opt.algo_str);
  printf("%ld flops for spmv\n", flops);

  sparse_t A = mk_sparse_random(opt.format, M, N, nnz, rg);
  sparse_t tA = sparse_transpose(A);
  vec_t x = mk_vec_unit_random(N, rg);
  vec_t y = mk_vec_zero(M);
  real lambda = repeat_spmv(opt.algo, A, tA, x, y, repeat);
  if (lambda == -1.0) {
    printf("an error ocurred during repeat_spmv\n");
  } else {
    printf("lambda = %.9e\n", lambda);
  }
  vec_destroy(x);
  vec_destroy(y);
  sparse_destroy(A);
  sparse_destroy(tA);
  cmdline_options_destroy(opt);
  return 0;
}

