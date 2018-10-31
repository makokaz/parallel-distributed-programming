#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clock.h"

/* GCC vector extension to define a vector of floats */
#if __AVX512F__
typedef float floatv __attribute__((vector_size(64)));
#else
typedef float floatv __attribute__((vector_size(32)));
#endif

/* vector size (SIMD lanes) */
const int vs = sizeof(floatv) / sizeof(float);

/** 
    @brief repeat x = a x + c for a scalar type (float) variable x
    @param (m) size of X. ignored. it always updates a single scalar element
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * vs floats)
    @param (c) c of a x + c
 */
long axpy_scalar(long n, floatv a, float* X, floatv c) {
  long i;
  float a0 = a[0], x0 = X[0], c0 = c[0];
  asm volatile ("# axpy_scalar: ax+c loop begin");
  for (i = 0; i < n; i++) {
    x0 = a0 * x0 + c0;
  }
  asm volatile ("# axpy_scalar: ax+c loop end");
  X[0] = x0;
  return 2 * n;
}

/** 
    @brief repeat x = a x + c for a vector (SIMD) type (floatv) variable x
    @param (m) size of X. ignored. it always updates an element
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * vs floats)
    @param (c) c of a x + c
 */
long axpy_simd(long n, floatv a, floatv* X, floatv c) {
  long i;
  floatv x = X[0];
  asm volatile ("# axpy_simd: ax+c loop begin");
  for (i = 0; i < n; i++) {
    x = a * x + c;
  }
  asm volatile ("# axpy_simd: ax+c loop end");
  X[0] = x;
  return 2 * vs * n;
}

/** 
    @brief repeat x = a x + c for a constant number of 
    vector type (floatv) variables
    @param (m) size of X. ignored. it always updates nv (constant) elements
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * vs floats)
    @param (c) c of a x + c
 */
template<int nv>
long axpy_simd_c(long n, floatv a, floatv* X, floatv c) {
  asm volatile ("# axpy_simd_c: ax+c loop begin");
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < nv; j++) {
      X[j] = a * X[j] + c;
    }
  }
  asm volatile ("# axpy_simd_c: ax+c loop end");
  return 2 * nv * vs * n;
}

/** 
    @brief repeat x = a x + c for m (variable) vector type (floatv) variables
    @param (m) the number of variables updated
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * vs floats)
    @param (c) c of a x + c

 */
long axpy_simd_m(long m, long n, floatv a, floatv* X, floatv c) {
  asm volatile ("# axpy_simd_m: ax+c loop begin");
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < m; j++) {
      X[j] = a * X[j] + c;
    }
  }
  asm volatile ("# axpy_simd_m: ax+c loop end");
  return 2 * m * vs * n;
}

/** 
    @brief repeat x = a x + c for m (variable) vector type (floatv) variables
    by updating a single variable a few times
    @param (m) the number of variables updated
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * vs floats)
    @param (c) c of a x + c

 */
long axpy_simd_m_nmn(long m, long n, floatv a, floatv* X, floatv c) {
  const int steps_inner = 4;
  asm volatile ("# axpy_simd_m_nmn: ax+c loop begin");
  for (long i = 0; i < n; i += steps_inner) {
    for (long j = 0; j < m; j++) {
      for (long ii = 0; ii < steps_inner; ii++) {
        X[j] = a * X[j] + c;
      }
    }
  }
  asm volatile ("# axpy_simd_m_nmn: ax+c loop end");
  return 2 * m * vs * (n - n % steps_inner);
}

/** 
    @brief repeat x = a x + c for m (variable) vector type (floatv) variables,
    nv (8) variables at a time

    @param (m) the number of variables updated
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * vs floats)
    @param (c) c of a x + c

 */
long axpy_simd_m_mnm(long m, long n, floatv a, floatv* X, floatv c) {
  const long nvars_inner = 8;
  for (long j = 0; j < m; j += nvars_inner) {
    asm volatile ("# axpy_simd_m_mnm: ax+c inner loop begin");
    for (long i = 0; i < n; i++) {
      for (long jj = 0; jj < nvars_inner; jj++) {
        X[j+jj] = a * X[j+jj] + c;
      }
    }
    asm volatile ("# axpy_simd_m_mnm: ax+c inner loop end");
  }
  return 2 * (m - m % nvars_inner) * vs * n;
}

/** 
    @brief repeat x = a x + c for m (variable) vector type (floatv) variables,
    nv (8) variables at a time

    @param (m) the number of variables updated
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * vs floats)
    @param (c) c of a x + c

 */
long axpy_simd_parallel_m_mnm(long m, long n, floatv a, floatv* X, floatv c) {
  const long nvars_inner = 8;
#pragma omp parallel for 
  for (long j = 0; j < m; j += nvars_inner) {
    asm volatile ("# axpy_simd_parallel_m_mnm: ax+c inner loop begin");
    for (long i = 0; i < n; i++) {
      for (long jj = 0; jj < nvars_inner; jj++) {
        X[j+jj] = a * X[j+jj] + c;
      }
    }
    asm volatile ("# axpy_simd_parallel_m_mnm: ax+c inner loop end");
  }
  return 2 * (m - m % nvars_inner) * vs * n;
}

/**
   @brief type of axpy functions
  */
typedef long (*axpy_t)(long m, long n, floatv a, floatv* X, floatv c);

typedef enum {
  algo_scalar, 
  algo_simd, 
  algo_simd_c, 
  algo_simd_m, 
  algo_simd_m_nmn, 
  algo_simd_m_mnm, 
  algo_simd_parallel_m_mnm, 
  algo_invalid,
} algo_t;

typedef struct {
  algo_t a;
  const char * name;
} algo_table_entry_t;

typedef struct {
  algo_table_entry_t t[algo_invalid + 1];
} algo_table_t;

const int max_template_m = 8;

/**
   @brief repeat a x + c by a specified algorithm
   @param (algo) algorithm
   @param (m) size of X. the actual number of elements used depends on algorithm
   @param (n) the number of times you do ax+c for each variable
   @param (a) a of a x + c
   @param (X) array of m floatv elements (i.e., m * vs floats)
   @param (c) c of a x + c
  */
long axpy(algo_t algo, long m, long n, floatv a, floatv* X, floatv c) {
  switch (algo) {
  case algo_scalar:
    return axpy_scalar(n, a, (float *)X, c);
  case algo_simd:
    return axpy_simd(n, a, X, c);
  case algo_simd_c: {
    switch (m) {
    case 1:
      return axpy_simd_c<1>(n, a, X, c);
    case 2:
      return axpy_simd_c<2>(n, a, X, c);
    case 3:
      return axpy_simd_c<3>(n, a, X, c);
    case 4:
      return axpy_simd_c<4>(n, a, X, c);
    case 5:
      return axpy_simd_c<5>(n, a, X, c);
    case 6:
      return axpy_simd_c<6>(n, a, X, c);
    case 7:
      return axpy_simd_c<7>(n, a, X, c);
    case 8:
      return axpy_simd_c<8>(n, a, X, c);
    default:
      assert(max_template_m == 8);
      fprintf(stderr, "algo_simd_const can take up to %d simd variables\n",
              max_template_m);
      return -1;
    }
  }
  case algo_simd_m:
    return axpy_simd_m(m, n, a, X, c);
  case algo_simd_m_nmn:
    return axpy_simd_m_nmn(m, n, a, X, c);
  case algo_simd_m_mnm:
    return axpy_simd_m_mnm(m, n, a, X, c);
  case algo_simd_parallel_m_mnm:
    return axpy_simd_parallel_m_mnm(m, n, a, X, c);
  default:
    fprintf(stderr, "invalid algorithm %d\n", algo);
    return -1;
  }
}

static algo_table_t algo_table = {
  {
    { algo_scalar, "scalar" },
    { algo_simd,   "simd" },
    { algo_simd_c, "simd_c" },
    { algo_simd_m, "simd_m" },
    { algo_simd_m_nmn, "simd_m_nmn" },
    { algo_simd_m_mnm, "simd_m_mnm" },
    { algo_simd_parallel_m_mnm, "simd_parallel_m_mnm" },
    { algo_invalid, 0 },
  }
};

algo_t parse_algo(char * s) {
  for (int i = 0; i < (int)algo_invalid; i++) {
    algo_table_entry_t e = algo_table.t[i];
    if (strcmp(e.name, s) == 0) {
      return e.a;
    }
  }
  fprintf(stderr, "%s:%d:parse_algo: invalid algo %s\n",
          __FILE__, __LINE__, s);
  return algo_invalid;
}

/**
   @brief main function
   @param (argc) the number of command line args
   @param (argv) command line args
  */
int main(int argc, char ** argv) {
  char * algo_str = (argc > 1 ? argv[1] : 0);
  algo_t     algo = (algo_str ? parse_algo(algo_str) : algo_scalar);
  if (algo == algo_invalid) return EXIT_FAILURE;
  long          m = (argc > 2 ? atol(argv[2]) : 8);
  long          n = (argc > 3 ? atol(argv[3]) : 100000000);
  long       seed = (argc > 4 ? atol(argv[4]) : 76843802738543);
  long n_elements_to_show = (argc > 5 ? atol(argv[5]) : 1);

  printf("algo = %s\n", algo_str);
  printf("m = %ld\n", m);
  printf("n = %ld\n", n);
  
  
  long   mm = m < max_template_m ? max_template_m : m;
  float a_[vs] __attribute__((aligned(64)));
  float X_[mm * vs] __attribute__((aligned(64)));
  float c_[vs] __attribute__((aligned(64)));
  unsigned short rg[3] = {
    (unsigned short)(seed >> 16),
    (unsigned short)(seed >> 8),
    (unsigned short)(seed) };
  for (int i = 0; i < vs; i++) {
    a_[i] = erand48(rg);
    c_[i] = erand48(rg);
  }
  for (int i = 0; i < mm * vs; i++) {
    X_[i] = erand48(rg);
  }
  floatv a = *((floatv*)a_);
  floatv * X = (floatv*)X_;
  floatv c = *((floatv*)c_);
  cpu_clock_counter_t cc = mk_cpu_clock_counter();
  long t0 = cur_time_ns();
  long c0 = cpu_clock_counter_get(cc);
  long long r0 = rdtsc();
  long flops = axpy(algo, m, n, a, X, c);
  long long r1 = rdtsc();
  long long c1 = cpu_clock_counter_get(cc);
  long t1 = cur_time_ns();
  long long dc = c1 - c0;
  long long dr = r1 - r0;
  long long dt = t1 - t0;
  if (flops == -1) {
    cpu_clock_counter_destroy(cc);
    return EXIT_FAILURE;
  } else {
    printf("%ld flops\n", flops);
    printf("%lld CPU clocks, %lld REF clocks, %lld ns\n", dc, dr, dt);
    printf("%f CPU clocks/iter, %f REF clocks/iter, %f ns/iter\n",
           dc / (double)n, dr / (double)n, dt / (double)n);
    printf("%f flops/CPU clock, %f flops/REF clock, %f GFLOPS\n",
           flops / (double)dc, flops / (double)dr, flops / (double)dt);
    for (int i = 0; i < n_elements_to_show; i++) {
      printf("x[%d] = %f\n", i, X_[i]);
    }
    cpu_clock_counter_destroy(cc);
    return EXIT_SUCCESS;
  }
}


