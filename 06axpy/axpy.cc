#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clock.h"

/* GCC vector extension to define a vector of floats */
#if __AVX512F__
const int vwidth = 64;
#elif __AVX__
const int vwidth = 32;
#else
#error "you'd better have a better machine"
#endif

const int valign = sizeof(float);
typedef float floatv __attribute__((vector_size(vwidth),aligned(valign)));
/* SIMD lanes */
const int L = sizeof(floatv) / sizeof(float);

/** 
    @brief repeat x = a x + c for a scalar type (float) variable x
    @param (m) size of X. ignored. it always updates a single scalar element
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * L floats)
    @param (c) c of a x + c

    @details it should run at 4 clocks/iter (the latency of fma
    instruction), or 0.5 flops/clock
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
    @param (X) array of m floatv elements (i.e., m * L floats)
    @param (c) c of a x + c

    @details it should run at 4 clocks/iter (the latency of fma
    instruction) = 4 flops/clock with AVX and 8 flops/clock with AVX512F 
 */
//#pragma GCC optimize("unroll-loops", 8)
long axpy_simd(long n, floatv a, floatv* X, floatv c) {
  long i;
  floatv x = X[0];
  asm volatile ("# axpy_simd: ax+c loop begin");
  for (i = 0; i < n; i++) {
    x = a * x + c;
  }
  asm volatile ("# axpy_simd: ax+c loop end");
  X[0] = x;
  return 2 * L * n;
}

/** 
    @brief repeat x = a x + c for a constant number of 
    vector type (floatv) variables
    @param (m) size of X. ignored. it always updates nv (constant) elements
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * L floats)
    @param (c) c of a x + c

    @details when you increase nv, it should remain running at 4 
    clocks/iter until it reaches the limit of 2 FMAs/cycle,
    where it achieves the peak performance. nv=8 should achieve
    64 flops/clock with AVX512F.
    
    $ srun -p big bash -c "./axpy simd_c 8"

    4.001386 CPU clocks/iter, 3.966710 REF clocks/iter, 1.893479 ns/iter
    63.977836 flops/CPU clock, 64.537118 flops/REF clock, 135.200880 GFLOPS

 */
template<int nv>
long axpy_simd_c(long _, long n, floatv a, floatv* X, floatv c) {
  (void)_;
  asm volatile ("# axpy_simd_c<%0>: ax+c loop begin" :: "g"(nv));
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < nv; j++) {
      X[j] = a * X[j] + c;
    }
  }
  asm volatile ("# axpy_simd_c<%0>: ax+c loop end" :: "g"(nv));
  return 2 * nv * L * n;
}

#if 0
template<int nv>
long axpy_simd_c(long _, long n, floatv a_, floatv* X_, floatv c_) {
  (void)_;
  float a = a_[0], c = c_[0];
  float * X = (float *)X_;
  asm volatile ("# axpy_simd_c<%0>: ax+c loop begin" :: "g"(nv));
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < nv * L; j++) {
      X[j] = a * X[j] + c;
    }
  }
  asm volatile ("# axpy_simd_c<%0>: ax+c loop end" :: "g"(nv));
  return 2 * nv * L * n;
}
#endif

/** 
    @brief repeat x = a x + c for m (variable) vector type (floatv) variables
    @param (m) the number of variables updated
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * L floats)
    @param (c) c of a x + c

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
long axpy_simd_m(long m, long n, floatv a, floatv* X, floatv c) {
  asm volatile ("# axpy_simd_m: ax+c loop begin");
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < m; j++) {
      X[j] = a * X[j] + c;
    }
  }
  asm volatile ("# axpy_simd_m: ax+c loop end");
  return 2 * m * L * n;
}

/** 
    @brief repeat x = a x + c for m (variable) vector type (floatv) variables
    by updating a single variable a few times
    @param (m) the number of variables updated
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * L floats)
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
  return 2 * m * L * (n - n % steps_inner);
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
template<int nv>
long axpy_simd_m_mnm(long m, long n, floatv a, floatv* X, floatv c) {
  for (long j = 0; j < m; j += nv) {
    asm volatile ("# axpy_simd_m_mnm<%0>: ax+c inner loop begin" :: "g"(nv));
    for (long i = 0; i < n; i++) {
      for (long jj = 0; jj < nv; jj++) {
        X[j+jj] = a * X[j+jj] + c;
      }
    }
    asm volatile ("# axpy_simd_m_mnm<%0>: ax+c inner loop end" :: "g"(nv));
  }
  return 2 * (m - m % nv) * L * n;
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
template<int nv>
long axpy_simd_parallel_m_mnm(long m, long n, floatv a, floatv* X, floatv c) {
#pragma omp parallel for schedule(static)
  for (long j = 0; j < m; j += nv) {
    floatv XX[nv];
    for (long jj = 0; jj < nv; jj++) {
      XX[jj] = X[j+jj];
    }
    asm volatile ("# axpy_simd_parallel_m_mnm<%0>: ax+c inner loop begin" :: "g"(nv));
    for (long i = 0; i < n; i++) {
      for (long jj = 0; jj < nv; jj++) {
        XX[jj] = a * XX[jj] + c;
      }
    }
    asm volatile ("# axpy_simd_parallel_m_mnm<%0>: ax+c inner loop end" :: "g"(nv));
    for (long jj = 0; jj < nv; jj++) {
      X[j+jj] = XX[jj];
    }
  }
  return 2 * (m - m % nv) * L * n;
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

typedef long (*axpy_fun_t)(long m, long n, floatv a, floatv* X, floatv c);
typedef struct {
  axpy_fun_t simd_c;
  axpy_fun_t simd_m_mnm;
  axpy_fun_t simd_parallel_m_mnm;
} axpy_funs_entry_t;

typedef struct {
  axpy_funs_entry_t t[20];
} axpy_funs_table_t;

axpy_funs_table_t axpy_funs_table = {
  {
    { 0, 0, 0, },
    { axpy_simd_c<1>, axpy_simd_m_mnm<1>,  axpy_simd_parallel_m_mnm<1>, },
    { axpy_simd_c<2>, axpy_simd_m_mnm<2>,  axpy_simd_parallel_m_mnm<2>, },
    { axpy_simd_c<3>, axpy_simd_m_mnm<3>,  axpy_simd_parallel_m_mnm<3>, },
    { axpy_simd_c<4>, axpy_simd_m_mnm<4>,  axpy_simd_parallel_m_mnm<4>, },
    { axpy_simd_c<5>, axpy_simd_m_mnm<5>,  axpy_simd_parallel_m_mnm<5>, },
    { axpy_simd_c<6>, axpy_simd_m_mnm<6>,  axpy_simd_parallel_m_mnm<6>, },
    { axpy_simd_c<7>, axpy_simd_m_mnm<7>,  axpy_simd_parallel_m_mnm<7>, },
    { axpy_simd_c<8>, axpy_simd_m_mnm<8>,  axpy_simd_parallel_m_mnm<8>, },
    { axpy_simd_c<9>,  axpy_simd_m_mnm<9>,   axpy_simd_parallel_m_mnm<9>, },
    { axpy_simd_c<10>, axpy_simd_m_mnm<10>,  axpy_simd_parallel_m_mnm<10>, },
    { axpy_simd_c<11>, axpy_simd_m_mnm<11>,  axpy_simd_parallel_m_mnm<11>, },
    { axpy_simd_c<12>, axpy_simd_m_mnm<12>,  axpy_simd_parallel_m_mnm<12>, },
    { axpy_simd_c<13>, axpy_simd_m_mnm<13>,  axpy_simd_parallel_m_mnm<13>, },
    { axpy_simd_c<14>, axpy_simd_m_mnm<14>,  axpy_simd_parallel_m_mnm<14>, },
    { axpy_simd_c<15>, axpy_simd_m_mnm<15>,  axpy_simd_parallel_m_mnm<15>, },
    { axpy_simd_c<16>, axpy_simd_m_mnm<16>,  axpy_simd_parallel_m_mnm<16>, },
    { axpy_simd_c<17>, axpy_simd_m_mnm<17>,  axpy_simd_parallel_m_mnm<17>, },
    { axpy_simd_c<18>, axpy_simd_m_mnm<18>,  axpy_simd_parallel_m_mnm<18>, },
    { axpy_simd_c<19>, axpy_simd_m_mnm<19>,  axpy_simd_parallel_m_mnm<19>, },
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
long axpy(algo_t algo, long nv, long m, long n, floatv a, floatv* X, floatv c) {
  int n_funs = sizeof(axpy_funs_table.t) / sizeof(axpy_funs_table.t[0]);
  if (nv < 1 || nv >= n_funs) {
    fprintf(stderr, "%s:%d:axpy: nv = %ld must be 1 < x < %d\n",
            __FILE__, __LINE__, nv, n_funs);
    return -1;
  }
  switch (algo) {
  case algo_scalar:
    return axpy_scalar(n, a, (float *)X, c);
  case algo_simd:
    return axpy_simd(n, a, X, c);
  case algo_simd_c: {
    axpy_t f = axpy_funs_table.t[nv].simd_c;
    return f(0, n, a, X, c);
  }
  case algo_simd_m:
    return axpy_simd_m(m, n, a, X, c);
  case algo_simd_m_nmn:
    return axpy_simd_m_nmn(m, n, a, X, c);
  case algo_simd_m_mnm: {
    axpy_t f = axpy_funs_table.t[nv].simd_m_mnm;
    return f(m, n, a, X, c);
  }
  case algo_simd_parallel_m_mnm: {
    axpy_t f = axpy_funs_table.t[nv].simd_parallel_m_mnm;
    return f(m, n, a, X, c);
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
  long         nv = (argc > 2 ? atol(argv[2]) : 8);
  long          m = (argc > 3 ? atol(argv[3]) : nv);
  long          n = (argc > 4 ? atol(argv[4]) : 1000000000);
  long       seed = (argc > 5 ? atol(argv[5]) : 76843802738543);
  long n_elements_to_show = (argc > 6 ? atol(argv[6]) : 1);

  if (m < nv) m = nv;
  
  printf(" algo = %s\n", algo_str);
  printf("    m = %ld\n", m);
  printf("    n = %ld\n", n);
  
  float a_[L]     __attribute__((aligned(64)));
  float X_[m * L] __attribute__((aligned(64)));
  float c_[L]     __attribute__((aligned(64)));
  unsigned short rg[3] = {
    (unsigned short)(seed >> 16),
    (unsigned short)(seed >> 8),
    (unsigned short)(seed) };
  for (int i = 0; i < L; i++) {
    a_[i] = erand48(rg);
    c_[i] = erand48(rg);
  }
  for (int i = 0; i < m * L; i++) {
    X_[i] = erand48(rg);
  }
  floatv a = *((floatv*)a_);
  floatv * X = (floatv*)X_;
  floatv c = *((floatv*)c_);
  cpu_clock_counter_t cc = mk_cpu_clock_counter();
  long t0 = cur_time_ns();
  long c0 = cpu_clock_counter_get(cc);
  long long r0 = rdtsc();
  long flops = axpy(algo, nv, m, n, a, X, c);
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
      printf("x[%d] = %f\n", i, X_[i]);
    }
    cpu_clock_counter_destroy(cc);
    return EXIT_SUCCESS;
  }
}


