/**
   @file axpy.cc
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "clock.h"

#if _OPENMP
#include <sched.h>
#include <omp.h>
#endif

/* GCC vector extension to define a vector of floats */
#if __AVX512F__
const int vwidth = 64;
#elif __AVX__
const int vwidth = 32;
#else
#error "you'd better have a better machine"
#endif

const int valign = vwidth;
//const int valign = sizeof(float);
typedef float floatv __attribute__((vector_size(vwidth),aligned(valign)));
/* SIMD lanes */
const int L = sizeof(floatv) / sizeof(float);

/** 
    @brief repeat x = a x + b for a scalar type (float) variable x
    @param (n) the number of times you do ax+b for x
    @param (a) a of a x + b
    @param (X) array of float elements (only use X[0])
    @param (b) b of a x + b

    @details it should run at 4 clocks/iter (the latency of fma
    instruction), or 0.5 flops/clock
 */
long axpy_scalar(long n, float a, float* X, float b) {
  long i;
  float x = X[0];
  asm volatile ("# axpy_scalar: ax+b loop begin");
  for (i = 0; i < n; i++) {
    x = a * x + b;
  }
  asm volatile ("# axpy_scalar: ax+b loop end");
  X[0] = x;
  return 2 * n;
}

/** 
    @brief repeat x = a x + b with SIMD instructions
    @param (n) the number of times you do ax+b
    @param (a) a of a x + b
    @param (X) array of float elements (use only L elements)
    @param (b) b of a x + b

    @details it should run at 4 clocks/iter (the latency of fma
    instruction) = 4 flops/clock with AVX and 8 flops/clock with AVX512F 
 */
//#pragma GCC optimize("unroll-loops", 4)
long axpy_simd(long n, float a, float* X_, float b) {
  long i;
  floatv * X = (floatv*)X_;
  floatv x = X[0];
  asm volatile ("# axpy_simd: ax+b loop begin");
  for (i = 0; i < n; i++) {
    x = a * x + b;
  }
  asm volatile ("# axpy_simd: ax+b loop end");
  X[0] = x;
  return 2 * L * n;
}

/** 
    @brief repeat x = a x + b for a constant number of 
    vector variables
    @param (m) size of X. ignored. it always updates c vector elements
    @param (n) the number of times you do ax+b for each variable
    @param (a) a of a x + b
    @param (X) array of float elements (use only c * L elements)
    @param (b) b of a x + b

    @details when you increase nv, it should remain running at 4 
    clocks/iter until it reaches the limit of 2 FMAs/cycle,
    where it achieves the peak performance. nv=8 should achieve
    64 flops/clock with AVX512F.
    
    $ srun -p big bash -c "./axpy simd_c 8"

    4.001386 CPU clocks/iter, 3.966710 REF clocks/iter, 1.893479 ns/iter
    63.977836 flops/CPU clock, 64.537118 flops/REF clock, 135.200880 GFLOPS
    
 */
template<int c>
long axpy_simd_c(long m, long n, float a, float* X_, float b) {
  assert(c <= m);
  assert(c % L == 0);
  floatv * X = (floatv*)X_;
  asm volatile ("# axpy_simd_c<%0>: ax+c loop begin" :: "i"(c));
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < c / L; j++) {
      X[j] = a * X[j] + b;
    }
  }
  asm volatile ("# axpy_simd_c<%0>: ax+c loop end" :: "i"(c));
  return 2 * c * n;
}

/** 
    @brief repeat x = a x + b for m (variable) vector type (floatv) variables
    @param (m) the number of variables updated
    @param (n) the number of times you do ax+b for each variable
    @param (a) a of a x + b
    @param (X) array of m float elements
    @param (b) b of a x + b

    @details this is similar to axpy_simc_c, but works on a variable
    number of vectors (m), which makes it impossible to completely
    unroll the j loop and therefore register-promote X.
    each innermost iteration therefore needs a load, an fma and a store
    instruction, which makes the latency longer and the throughput
    limited by the throughput of store instructions.
    
    $ srun -p big bash -c "./axpy simd_m 8"
    algo = simd_m
    m = 8
    n = 100000000
    flops = 25600000000
    1802238053 CPU clocks, 1397861786 REF clocks, 667250569 ns
    18.022381 CPU clocks/iter, 13.978618 REF clocks/iter, 6.672506 ns/iter
    14.204561 flops/CPU clock, 18.313685 flops/REF clock, 38.366397 GFLOPS

 */
long axpy_simd_m(long m, long n, float a, float * X_, float b) {
  assert(m % L == 0);
  floatv * X = (floatv*)X_;
  asm volatile ("# axpy_simd_m: ax+c loop begin");
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < m / L; j++) {
      X[j] = a * X[j] + b;
    }
  }
  asm volatile ("# axpy_simd_m: ax+c loop end");
  return 2 * m * n;
}

/** 
    @brief repeat x = a x + b for m (variable) vector type (floatv) variables
    by updating a single variable a few times
    @param (m) the number of variables updated
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + b
    @param (X) array of m floatv elements (i.e., m * L floats)
    @param (b) b of a x + b

 */
long axpy_simd_m_nmn(long m, long n, float a, float* X_, float b) {
  assert(m % L == 0);
  floatv * X = (floatv*)X_;
  const int steps_inner = 4;
  asm volatile ("# axpy_simd_m_nmn: ax+b loop begin");
  for (long i = 0; i < n; i += steps_inner) {
    for (long j = 0; j < m / L; j++) {
      for (long ii = 0; ii < steps_inner; ii++) {
        X[j] = a * X[j] + b;
      }
    }
  }
  asm volatile ("# axpy_simd_m_nmn: ax+b loop end");
  return 2 * m * (n - n % steps_inner);
}

/** 
    @brief repeat x = a x + c for m (variable) vector type (floatv) variables,
    nv variables at a time

    @param (m) the number of variables updated
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * L floats)
    @param (c) c of a x + c

    @details the innsermost two loops look similar to axpy_simd_c

 */
template<int c>
long axpy_simd_m_mnm(long m, long n, float a, float * X_, float b) {
  assert(m % c == 0);
  assert(c % L == 0);
  floatv * X = (floatv*)X_;
  for (long j = 0; j < m / L; j += c / L) {
    asm volatile ("# axpy_simd_m_mnm<%0>: ax+c inner loop begin" :: "i"(c));
    for (long i = 0; i < n; i++) {
      for (long jj = 0; jj < c / L; jj++) {
        X[j+jj] = a * X[j+jj] + b;
      }
    }
    asm volatile ("# axpy_simd_m_mnm<%0>: ax+c inner loop end" :: "i"(c));
  }
  return 2 * m * n;
}

/** 
    @brief repeat x = a x + c for m (variable) vector type (floatv) variables in parallel,
    nv variables at a time

    @param (m) the number of variables updated
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * L floats)
    @param (c) c of a x + c

    @details
    $ srun -p big -n 1 --exclusive bash -c "OMP_PROC_BIND=true OMP_NUM_THREADS=64 ./axpy simd_parallel_m_mnm 8 512 100000000"
    should achieve something like this on the big partition
    4.125885 CPU clocks/iter, 4.708529 REF clocks/iter, 2.247610 ns/iter
    3971.026909 flops/CPU clock, 3479.643183 flops/REF clock, 7289.520058 GFLOPS

 */
template<int c>
long axpy_simd_parallel_m_mnm(long m, long n, float a, float * X__, float b) {
  assert(c % L == 0);
  assert(m % c == 0);
  floatv * X_ = (floatv*)X__;
#pragma omp parallel for schedule(static)
  for (long j = 0; j < m / L; j += c / L) {
    floatv X[c/L];
    for (long jj = 0; jj < c / L; jj++) {
      X[jj] = X_[j+jj];
    }
    asm volatile ("# axpy_simd_parallel_m_mnm<%0>: ax+c inner loop begin" :: "i"(c));
    for (long i = 0; i < n; i++) {
      for (long jj = 0; jj < c / L; jj++) {
        X[jj] = a * X[jj] + b;
      }
    }
    asm volatile ("# axpy_simd_parallel_m_mnm<%0>: ax+c inner loop end" :: "i"(c));
    for (long jj = 0; jj < c / L; jj++) {
      X_[j+jj] = X[jj];
    }
  }
  return 2 * m * n;
}

/**
   @brief type of axpy functions
  */

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

typedef long (*axpy_fun_t)(long m, long n, float a, float* X, float c);
typedef struct {
  axpy_fun_t simd_c;
  axpy_fun_t simd_m_mnm;
  axpy_fun_t simd_parallel_m_mnm;
} axpy_funs_entry_t;

typedef struct {
  axpy_funs_entry_t t[50];
} axpy_funs_table_t;

#define mk_ent(c) { axpy_simd_c<c*L>, axpy_simd_m_mnm<c*L>,  axpy_simd_parallel_m_mnm<c*L>, }

axpy_funs_table_t axpy_funs_table = {
  {
    { 0, 0, 0, }, mk_ent(1),  mk_ent(2),  mk_ent(3),  mk_ent(4),
    mk_ent(5),    mk_ent(6),  mk_ent(7),  mk_ent(8),  mk_ent(9),
    mk_ent(10),   mk_ent(11), mk_ent(12), mk_ent(13), mk_ent(14),
    mk_ent(15),   mk_ent(16), mk_ent(17), mk_ent(18), mk_ent(19),
    mk_ent(20),   mk_ent(21), mk_ent(22), mk_ent(23), mk_ent(24),
    mk_ent(25),   mk_ent(26), mk_ent(27), mk_ent(28), mk_ent(29),
    mk_ent(30),   mk_ent(31), mk_ent(32), mk_ent(33), mk_ent(34),
    mk_ent(35),   mk_ent(36), mk_ent(37), mk_ent(38), mk_ent(39),
    mk_ent(40),   mk_ent(41), mk_ent(42), mk_ent(43), mk_ent(44),
    mk_ent(45),   mk_ent(46), mk_ent(47), mk_ent(48), mk_ent(49),
  }
};

/**
   @brief repeat a x + c by a specified algorithm
   @param (algo) algorithm
   @param (m) size of X. the actual number of elements used depends on algorithm
   @param (n) the number of times you do ax+c for each variable
   @param (a) a of a x + c
   @param (X) array of m floatv elements (i.e., m * L floats)
   @param (c) c of a x + c
  */
long axpy(algo_t algo, long c, long m, long n, float a, float* X, float b) {
  int n_funs = sizeof(axpy_funs_table.t) / sizeof(axpy_funs_table.t[0]);
  assert(c % L == 0);
  long idx = c / L;
  if (idx < 1 || idx >= n_funs) {
    fprintf(stderr, "%s:%d:axpy: c = %ld must be %d < c < %d\n",
            __FILE__, __LINE__, c, L, n_funs * L);
    return -1;
  }
  switch (algo) {
  case algo_scalar:
    return axpy_scalar(n, a, X, b);
  case algo_simd:
    return axpy_simd(n, a, X, b);
  case algo_simd_c: {
    axpy_fun_t f = axpy_funs_table.t[idx].simd_c;
    return f(m, n, a, X, b);
  }
  case algo_simd_m:
    return axpy_simd_m(m, n, a, X, b);
  case algo_simd_m_nmn:
    return axpy_simd_m_nmn(m, n, a, X, b);
  case algo_simd_m_mnm: {
    axpy_fun_t f = axpy_funs_table.t[idx].simd_m_mnm;
    return f(m, n, a, X, b);
  }
  case algo_simd_parallel_m_mnm: {
    axpy_fun_t f = axpy_funs_table.t[idx].simd_parallel_m_mnm;
    return f(m, n, a, X, b);
  }
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
  char * algo_str = (argc > 1 ? argv[1] : (char *)"scalar");
  algo_t     algo = (algo_str ? parse_algo(algo_str) : algo_scalar);
  if (algo == algo_invalid) return EXIT_FAILURE;
  long          c = (argc > 2 ? atol(argv[2]) : 8);
  long          m = (argc > 3 ? atol(argv[3]) : c);
  long          n = (argc > 4 ? atol(argv[4]) : 1000000000);
  long       seed = (argc > 5 ? atol(argv[5]) : 76843802738543);
  long n_elements_to_show = (argc > 6 ? atol(argv[6]) : 1);

  if (m < c) m = c;
  if (n_elements_to_show >= m) n_elements_to_show = m;
  
  printf(" algo = %s\n", algo_str);
  printf("    c = %ld (the number of variables to update in the inner loop)\n", c);
  printf("    m = %ld (the total number of variables to update)\n", m);
  printf("    n = %ld (the number of times to update each variable)\n", n);
  
  unsigned short rg[3] = {
    (unsigned short)(seed >> 16),
    (unsigned short)(seed >> 8),
    (unsigned short)(seed) };
  float a = erand48(rg);
  float b = erand48(rg);
  float X[m] __attribute__((aligned(64)));
  for (int i = 0; i < m; i++) {
    X[i] = erand48(rg);
  }
  cpu_clock_counter_t cc = mk_cpu_clock_counter();
  long t0 = cur_time_ns();
  long c0 = cpu_clock_counter_get(cc);
  long long r0 = rdtsc();
  long flops = axpy(algo, c, m, n, a, X, b);
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
    printf("flops = %ld\n", flops);
    printf("%lld CPU clocks, %lld REF clocks, %lld ns\n", dc, dr, dt);
    printf("%f CPU clocks/iter, %f REF clocks/iter, %f ns/iter\n",
           dc / (double)n, dr / (double)n, dt / (double)n);
    printf("%f flops/CPU clock, %f flops/REF clock, %f GFLOPS\n",
           flops / (double)dc, flops / (double)dr, flops / (double)dt);
    for (int i = 0; i < n_elements_to_show; i++) {
      printf("x[%d] = %f\n", i, X[i]);
    }
    cpu_clock_counter_destroy(cc);
    return EXIT_SUCCESS;
  }
}


