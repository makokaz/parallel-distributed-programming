#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clock.h"

/* GCC vector extension to define a vector of floats */
typedef float floatv __attribute__((vector_size(64)));
/* vector size (SIMD lanes) */
const int vs = sizeof(floatv) / sizeof(float);
/* the number of vector variables to update concurrently
   to reach the maximum throughput.  it is 2 * latency
   of a single fma */
const int nv = 8;

/** 
    @brief repeat x = a x + c for a scalar type (float) variable x
    @param (m) size of X. ignored. it always updates a single scalar element
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * vs floats)
    @param (c) c of a x + c
 */
long axpy_0(long m, long n, floatv a, floatv X[m], floatv c) {
  long i;
  floatv x = X[0];
  float a0 = a[0], x0 = x[0], c0 = c[0];
  asm volatile ("# axpy_0: ax+c loop begin");
  for (i = 0; i < n; i++) {
    x0 = a0 * x0 + c0;
  }
  asm volatile ("# axpy_0: ax+c loop end");
  ((float *)X)[0] = x0;
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
long axpy_1(long m, long n, floatv a, floatv X[m], floatv c) {
  long i;
  floatv x = X[0];
  asm volatile ("# axpy_1: ax+c loop begin");
  for (i = 0; i < n; i++) {
    x = a * x + c;
  }
  asm volatile ("# axpy_1: ax+c loop end");
  X[0] = x;
  return 2 * vs * n;
}

/** 
    @brief repeat x = a x + c for nv (8) vector type (floatv) variables
    @param (m) size of X. ignored. it always updates nv (constant) elements
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * vs floats)
    @param (c) c of a x + c
 */
long axpy_2(long m, long n, floatv a, floatv X[m], floatv c) {
  asm volatile ("# axpy_2: ax+c loop begin");
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < nv; j++) {
      X[j] = a * X[j] + c;
    }
  }
  asm volatile ("# axpy_2: ax+c loop end");
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
long axpy_3(long m, long n, floatv a, floatv X[m], floatv c) {
  asm volatile ("# axpy_3: ax+c loop begin");
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < m; j++) {
      X[j] = a * X[j] + c;
    }
  }
  asm volatile ("# axpy_3: ax+c loop end");
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
long axpy_4(long m, long n, floatv a, floatv X[m], floatv c) {
  const int unroll_factor = 4;
  asm volatile ("# axpy_4: ax+c loop begin");
  for (long i = 0; i < n; i += unroll_factor) {
    for (long j = 0; j < m; j++) {
      for (long ii = 0; ii < unroll_factor; ii++) {
        X[j] = a * X[j] + c;
      }
    }
  }
  asm volatile ("# axpy_4: ax+c loop end");
  return 2 * m * vs * (n - n % unroll_factor);
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
long axpy_5(long m, long n, floatv a, floatv X[m], floatv c) {
  asm volatile ("# axpy_5: ax+c loop begin");
  for (long j = 0; j < m; j += nv) {
    for (long i = 0; i < n; i++) {
      for (long jj = 0; jj < nv; jj++) {
        X[j+jj] = a * X[j+jj] + c;
      }
    }
  }
  asm volatile ("# axpy_5: ax+c loop end");
  return 2 * (m - m % nv) * vs * n;
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
long axpy_6(long m, long n, floatv a, floatv X[m], floatv c) {
#pragma omp parallel for 
  for (long j = 0; j < m; j += nv) {
    for (long i = 0; i < n; i++) {
      for (long jj = 0; jj < nv; jj++) {
        X[j+jj] = a * X[j+jj] + c;
      }
    }
  }
  return 2 * (m - m % nv) * vs * n;
}

/**
   @brief type of axpy functions
  */
typedef long (*axpy_t)(long m, long n, floatv a, floatv X[m], floatv c);

/**
   @brief repeat a x + c by a specified algorithm
   @param (algo) algorithm
   @param (m) size of X. the actual number of elements used depends on algorithm
   @param (n) the number of times you do ax+c for each variable
   @param (a) a of a x + c
   @param (X) array of m floatv elements (i.e., m * vs floats)
   @param (c) c of a x + c
  */
long axpy(long algo, long m, long n, floatv a, floatv X[m], floatv c) {
  axpy_t funs[] = { axpy_0, axpy_1, axpy_2, axpy_3, axpy_4, axpy_5, axpy_6 };
  int n_algos = sizeof(funs) / sizeof(axpy_t);
  assert(algo < n_algos);
  return funs[algo](m, n, a, X, c);
}

/**
   @brief main function
   @param (argc) the number of command line args
   @param (argv) command line args
  */
int main(int argc, char ** argv) {
  long algo = (argc > 1 ? atol(argv[1]) : 0);
  long    m = (argc > 2 ? atol(argv[2]) : 8);
  long    n = (argc > 3 ? atol(argv[3]) : 100000000);
  long seed = (argc > 4 ? atol(argv[4]) : 76843802738543);
  long n_elements_to_show = (argc > 5 ? atol(argv[5]) : 1);

  printf("algo = %ld\n", algo);
  long   mm = m < nv ? nv : m;
  float a_[vs] __attribute__((aligned(64)));
  float X_[mm * vs] __attribute__((aligned(64)));
  float c_[vs] __attribute__((aligned(64)));
  unsigned short rg[3] = { seed >> 16, seed >> 8, seed };
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
  return 0;
}


