/* 
 * mmc.cc
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "clock.h"
#include MMC_H

#include <x86intrin.h>

template<int M,int N,int K,int lda,int ldb>
static float comp_ij(matrix_c<M,K,lda>& A, matrix_c<K,N,ldb>& B,
                     long i, long j, long times) {
  float s = 0.0;
  //long K = A.nC;
  for (long t = 0; t < times; t++) {
    asm volatile("# comp_ij K loop begins");
    for (long k = 0; k < K; k++) {
      s += A(i,k) * B(k,j);
    }
    asm volatile("# comp_ij K loop ends");
  }
  return s;
}

static char * wipe_cache(int x) {
  static char * a = 0;
  long n = 1000 * 1000 * 1000;
  if (!a) a = (char *)malloc(n);
  memset(a, x, n);
  return a;
}

int main(int argc, char ** argv) {
  long times = (argc > 1 ? atol(argv[1]) : 100000);
  long chk   = (argc > 2 ? atol(argv[2]) : 1);
  long seed  = (argc > 3 ? atol(argv[3]) : 76843802738543);

  const long dM = 6;
  const long dN = 2;
  const long nV = dM * dN;
  const long M = nV * 1;
  const long N = 32;
  const long K = 192;
  const long lda = K;
  const long ldb = N;
  const long ldc = N;
  
  assert(K <= lda);
  assert(N <= ldb);
  assert(N <= ldc);
  //assert(M % dM == 0);
  //assert(N % (dN * L) == 0);

  matrix_c<M,K,lda> A;
  matrix_c<K,N,ldb> B;
  matrix_c<M,N,ldc> C;

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
  printf("A : %ld x %ld (ld=%ld) %ld bytes\n",
         M, K, lda, M * K * sizeof(float));
  printf("B : %ld x %ld (ld=%ld) %ld bytes\n",
         K, N, ldb, K * N * sizeof(float));
  printf("C : %ld x %ld (ld=%ld) %ld bytes\n",
         M, N, ldc, M * N * sizeof(float));
  printf("total = %ld bytes\n",
	 (M * K + K * N + M * N) * sizeof(float));
  char * wipe = wipe_cache(0);
  printf("repeat : %ld times\n", times);
  printf("perform %ld flops ... ", flops_all); fflush(stdout);
  cpu_clock_counter_t cc = mk_cpu_clock_counter();

  long n_iters = 0;
  long long t0 = cur_time_ns();
  long long c0 = cpu_clock_counter_get(cc);
  long long r0 = rdtsc();
  for (int i = 0; i < times; i++) {
    n_iters += gemm<M,N,K,lda,ldb,ldc,nV,dN>(A, B, C);
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
    printf("C(%ld,%ld) = %f, ans = %f, |C(%ld,%ld) - s| = %.9f\n",
	   i, j, C(i,j), s, i, j, fabs(C(i,j) - s));
  }
  if (wipe) free(wipe);
  cpu_clock_counter_destroy(cc);
  return 0;
}
