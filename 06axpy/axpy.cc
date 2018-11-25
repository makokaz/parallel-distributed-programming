/**
   @file axpy.cc
 */

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <x86intrin.h>

#include "clock.h"

#if _OPENMP
#include <sched.h>
#include <omp.h>
#endif

#if __NVCC__
/* cuda_util.h incudes various utilities to make CUDA 
   programming less error-prone. check it before you
   proceed with rewriting it for CUDA */
#include "include/cuda_util.h"
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
  algo_cuda,
  algo_invalid,
} algo_t;

/**
   @brief command line options
 */

typedef struct {
  const char * algo_str;
  algo_t algo;
  long b;                       /**< cuda block size */
  long c;                       /**< the number of floats concurrently updated */
  long m;                       /**< the number of floats */
  long n;                       /**< the number of times each variable is updated */
  long seed;                    /**< random seed */
  long n_elems_to_show;         /**< the number of variables to show results */
  int help;
  int error;
} cmdline_options_t;

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
    @brief repeat x = a x + b for m (variable) vector type (floatv) variables in parallel,
    nv variables at a time

    @param (m) the number of variables updated
    @param (n) the number of times you do ax+b for each variable
    @param (a) a of a x + b
    @param (X) array of m floatv elements (i.e., m * L floats)
    @param (b) b of a x + b

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
    @brief repeat x = a x + b for m (variable) vector type (floatv) variables in parallel,
    nv variables at a time

    @param (m) the number of variables updated
    @param (n) the number of times you do ax+b for each variable
    @param (a) a of a x + b
    @param (X) array of m floatv elements (i.e., m * L floats)
    @param (b) b of a x + b

    @details
    $ srun -p big -n 1 --exclusive bash -c "OMP_PROC_BIND=true OMP_NUM_THREADS=64 ./axpy simd_parallel_m_mnm 8 512 100000000"
    should achieve something like this on the big partition
    4.125885 CPU clocks/iter, 4.708529 REF clocks/iter, 2.247610 ns/iter
    3971.026909 flops/CPU clock, 3479.643183 flops/REF clock, 7289.520058 GFLOPS

 */

#if __NVCC__
__global__ void axpy_dev(long m, long n, float a, float * X, float b) {
  int j = get_thread_id_x();
  if (j < m) {
    for (long i = 0; i < n; i++) {
      X[j] = a * X[j] + b;
    }
  }
}

long axpy_cuda(long m, long n, float a, float * X, float b) {
  size_t sz = sizeof(float) * m;
  float * X_dev = (float *)dev_malloc(sz);
  to_dev(X_dev, X, sz);
  int bs = 512;
  int nb = (m + bs - 1) / bs;
  check_launch_error((axpy_dev<<<nb,bs>>>(m, n, a, X_dev, b)));
  to_host(X, X_dev, sz);
  dev_free(X_dev);
  return 2 * m * n;
}
#endif

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
long axpy(cmdline_options_t opt, float a, float* X, float b) {
  int n_funs = sizeof(axpy_funs_table.t) / sizeof(axpy_funs_table.t[0]);
  long idx = opt.c / L;
  switch (opt.algo) {
  case algo_simd_c:
  case algo_simd_m_mnm:
  case algo_simd_parallel_m_mnm: {
    if (idx < 1 || idx >= n_funs) {
      fprintf(stderr, "%s:%d:axpy: c = %ld must be %d < c < %d\n",
              __FILE__, __LINE__, opt.c, L, n_funs * L);
      return -1;
    }
    break;
  }
  default:
    break;
  }
  switch (opt.algo) {
  case algo_scalar:
    return axpy_scalar(opt.n, a, X, b);
  case algo_simd:
    return axpy_simd(opt.n, a, X, b);
  case algo_simd_c: {
    axpy_fun_t f = axpy_funs_table.t[idx].simd_c;
    return f(opt.m, opt.n, a, X, b);
  }
  case algo_simd_m:
    return axpy_simd_m(opt.m, opt.n, a, X, b);
  case algo_simd_m_nmn:
    return axpy_simd_m_nmn(opt.m, opt.n, a, X, b);
  case algo_simd_m_mnm: {
    axpy_fun_t f = axpy_funs_table.t[idx].simd_m_mnm;
    return f(opt.m, opt.n, a, X, b);
  }
  case algo_simd_parallel_m_mnm: {
    axpy_fun_t f = axpy_funs_table.t[idx].simd_parallel_m_mnm;
    return f(opt.m, opt.n, a, X, b);
  }
#if __NVCC__
  case algo_cuda: {
    return axpy_cuda(opt.m, opt.n, a, X, b);
  }
#endif
  default:
    fprintf(stderr, "invalid algorithm %s\n", opt.algo_str);
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
    { algo_cuda, "cuda" },
    { algo_invalid, 0 },
  }
};

algo_t parse_algo(const char * s) {
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

static cmdline_options_t default_opts() {
  cmdline_options_t opt = {
    .algo_str = "scalar",
    .algo = algo_invalid,
    .b = 512,
    .c = L,
    .m = -1,                     // = c
    .n = 1000000000,
    .seed = 76843802738543,
    .n_elems_to_show = 1,
    .help = 0,
    .error = 0,
  };
  return opt;
}

static void usage(const char * prog) {
  cmdline_options_t o = default_opts();
  fprintf(stderr,
          "usage:\n"
          "\n"
          "  %s [options ...]\n"
          "\n"
          "options:\n"
          "  --help                  show this help\n"
          "  -a,--algo A             use algorithm A (scalar,simd,simd_c,simd_m,simd_mnm,simd_nmn,simd_parallel_mnm,cuda) [%s]\n"
          "  -b,--cuda-block-size N  set cuda block size to N [%ld]\n"
          "  -c,--concurrent-vars N  concurrently update N floats [%ld]\n"
          "  -m,--vars N             update N floats [%ld]\n"
          "  -n,--n N                update each float variable N times [%ld]\n"
          "  -s,--seed N             set random seed to N [%ld]\n"
          ,
          prog,
          o.algo_str,
          o.b, o.c, o.m, o.n, o.seed
          );
}

/** 
    @brief command line options
*/
static struct option long_options[] = {
  {"algo",            required_argument, 0, 'a' },
  {"cuda-block-size", required_argument, 0, 'b' },
  {"concurrent-vars", required_argument, 0, 'c' },
  {"vars",            required_argument, 0, 'm' },
  {"n",               required_argument, 0, 'n' },
  {"seed",            required_argument, 0, 's' },
  {"help",            no_argument,       0, 'h'},
  {0,                 0,                 0,  0 }
};

/**

 */
static cmdline_options_t parse_args(int argc, char ** argv) {
  char * prog = argv[0];
  cmdline_options_t opt = default_opts();
  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "a:b:c:m:n:s:h",
                        long_options, &option_index);
    if (c == -1) break;
    switch (c) {
    case 0:
      {
        const char * o = long_options[option_index].name;
        fprintf(stderr,
                "bug:%s:%d: should handle option %s\n",
                __FILE__, __LINE__, o);
        opt.error = 1;
        return opt;
      }
      break;
    case 'a':
      opt.algo_str = strdup(optarg);
      break;
    case 'b':
      opt.b = atol(optarg);
      break;
    case 'c':
      opt.c = atol(optarg);
      break;
    case 'm':
      opt.m = atol(optarg);
      break;
    case 'n':
      opt.n = atol(optarg);
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
  opt.algo = parse_algo(opt.algo_str);
  if (opt.algo == algo_invalid) {
    opt.error = 1;
    return opt;
  }
  if (opt.m < opt.c) {
    opt.m = opt.c;
  }
  if (opt.n_elems_to_show >= opt.m) {
    opt.n_elems_to_show = opt.m;
  }
  return opt;
}

/**
   @brief main function
   @param (argc) the number of command line args
   @param (argv) command line args
  */
int main(int argc, char ** argv) {
  cmdline_options_t opt = parse_args(argc, argv);
  if (opt.help || opt.error) {
    usage(argv[0]);
    exit(opt.error);
  }

  printf(" algo = %s\n", opt.algo_str);
  printf("    b = %ld (cuda block size)\n", opt.b);
  printf("    c = %ld (the number of variables to update in the inner loop)\n", opt.c);
  printf("    m = %ld (the total number of variables to update)\n", opt.m);
  printf("    n = %ld (the number of times to update each variable)\n", opt.n);
  
  unsigned short rg[3] = {
    (unsigned short)(opt.seed >> 16),
    (unsigned short)(opt.seed >> 8),
    (unsigned short)(opt.seed) };
  float a = erand48(rg);
  float b = erand48(rg);
  //float X[opt.m] __attribute__((aligned(64)));
  float * X = (float *)_mm_malloc(sizeof(float) * opt.m, 64);
  for (int i = 0; i < opt.m; i++) {
    X[i] = erand48(rg);
  }
  cpu_clock_counter_t cc = mk_cpu_clock_counter();
  long t0 = cur_time_ns();
  long c0 = cpu_clock_counter_get(cc);
  long long r0 = rdtsc();
  long flops = axpy(opt, a, X, b);
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
           dc / (double)opt.n, dr / (double)opt.n, dt / (double)opt.n);
    printf("%f flops/CPU clock, %f flops/REF clock, %f GFLOPS\n",
           flops / (double)dc, flops / (double)dr, flops / (double)dt);
    for (int i = 0; i < opt.n_elems_to_show; i++) {
      printf("x[%d] = %f\n", i, X[i]);
    }
    cpu_clock_counter_destroy(cc);
    return EXIT_SUCCESS;
  }
}


