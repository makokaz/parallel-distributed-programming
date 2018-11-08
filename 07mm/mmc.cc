/* 
 * mm.cc
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "clock.h"
#include "mm.h"

#include <x86intrin.h>

long make_multiple(long n, long m) {
  n = n + m - 1;
  return n - (n % m);
}

long gcd_rec(long a, long b) {
  if (b == 0) return a;
  else {
    return gcd_rec(b, a % b);
  }
}

long gcd(long a, long b) {
  assert(b >= 0);
  if (a >= b) {
    return gcd_rec(a, b);
  } else {
    return gcd_rec(b, a);
  }
}

long make_padding(long n0, long m, long p) {
  long n = make_multiple(n0, m);
  assert(p % m == 0);
  while (gcd(n, p) != m) {
    n += m;
  }
  return n;
}

float comp_ij(matrix& A, matrix& B, long i, long j, long times) {
  float s = 0.0;
  long K = A.nC;
  for (long t = 0; t < times; t++) {
    for (long k = 0; k < K; k++) {
      s += A(i,k) * B(k,j);
    }
  }
  return s;
}

char * wipe_cache(int x) {
  static char * a = 0;
  long n = 1000 * 1000 * 1000;
  if (!a) a = (char *)malloc(n);
  memset(a, x, n);
  return a;
}

long align16(long x) {
  return (x + 15) & ~15;
}

int main(int argc, char ** argv) {
  long M    = (argc > 1 ? atol(argv[1]) : 8);
  long N    = (argc > 2 ? atol(argv[2]) : 32);
  long K    = (argc > 3 ? atol(argv[3]) : 192);
  long lda_ = (argc > 4 ? atol(argv[4]) : 0);
  long ldb_ = (argc > 5 ? atol(argv[5]) : 0);
  long ldc_ = (argc > 6 ? atol(argv[6]) : 0);
  long times = (argc > 7 ? atol(argv[7]) : 100000);
  long chk  = (argc > 8 ? atol(argv[8]) : 1);
  long seed = (argc > 9 ? atol(argv[9]) : 76843802738543);
  long lda = (lda_ ? lda_ : K);
  long ldb = (ldb_ ? ldb_ : N);
  long ldc = (ldc_ ? ldc_ : N);

  long a_sz = align16(M * lda);
  long b_sz = align16(K * ldb);
  long c_sz = align16(M * ldc);
#if 1
  float * abc = (float *)alloc64(sizeof(float) * (a_sz + b_sz + c_sz));
  float * a = abc;
  float * b = &a[a_sz];
  float * c = &b[b_sz];
#else
  float * a = (float *)alloc64(sizeof(float) * a_sz);
  float * b = (float *)alloc64(sizeof(float) * b_sz);
  float * c = (float *)alloc64(sizeof(float) * c_sz);
#endif
  matrix A(M, K, lda, a);
  matrix B(K, N, ldb, b);
  matrix C(M, N, ldc, c);
  unsigned short rg[3] = { (unsigned short)((seed >> 16) & 65535),
			   (unsigned short)((seed >> 8)  & 65535),
			   (unsigned short)((seed >> 0)  & 65535) };
  long fmas      = M * N * K;
  long flops     = 2 * fmas;
  long flops_all = flops * times;
  A.rand_init(rg);
  B.rand_init(rg);
  C.zero();
  printf("M = %ld, N = %ld, K = %ld\n", M, N, K);
  printf("A : %ld x %ld (ld=%ld) %ld bytes\n", A.nR, A.nC, A.ld, A.nR * A.nC * sizeof(float));
  printf("B : %ld x %ld (ld=%ld) %ld bytes\n", B.nR, B.nC, B.ld, B.nR * B.nC * sizeof(float));
  printf("C : %ld x %ld (ld=%ld) %ld bytes\n", C.nR, C.nC, C.ld, C.nR * C.nC * sizeof(float));
  printf("total = %ld bytes\n",
	 (A.nR * A.nC + B.nR * B.nC + C.nR * C.nC) * sizeof(float));
  char * wipe = wipe_cache(0);
  printf("repeat : %ld times\n", times);
  printf("perform %ld flops ... ", flops_all); fflush(stdout);
  cpu_clock_counter_t cc = mk_cpu_clock_counter();

  long n_iters = 0;
  long long t0 = cur_time_ns();
  long long c0 = cpu_clock_counter_get(cc);
  long long r0 = rdtsc();
  for (int i = 0; i < times; i++) {
    n_iters += gemm(A, B, C);
  }
  long long r1 = rdtsc();
  long long c1 = cpu_clock_counter_get(cc);
  long long t1 = cur_time_ns();
  long long dr = r1 - r0;
  long long dc = c1 - c0;
  long long dt = t1 - t0;

  printf("done \n");
  printf("%lld CPU clocks\n", dc);
  printf("%lld REF clocks\n", dr);
  printf("%lld nano sec\n", dt);
  printf("%.3f CPU clocks/iter\n",  dc / (double)n_iters);
  printf("%.3f REF clocks/iter\n",  dr / (double)n_iters);
  printf("%.3f flops/CPU clock\n", flops_all / (double)dc);
  printf("%.3f flops/REF clock\n", flops_all / (double)dr);
  printf("%.3f GFLOPS\n",          flops_all / (double)dt);

  if (chk) {
    long i = nrand48(rg) % M;
    long j = nrand48(rg) % N;
    float s = comp_ij(A, B, i, j, times);
    printf("C(%ld,%ld) = %f, ans = %f, |C(%ld,%ld) - s| = %f\n",
	   i, j, C(i,j), s, i, j, fabs(C(i,j) - s));
  }
  if (wipe) free(wipe);
  cpu_clock_counter_destroy(cc);
  return 0;
}
