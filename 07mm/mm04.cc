/* 
 * mm5.cc
 */

#include "mm.h"

/* similar to mm_4.h, but update 5 x 2 tiles of C
   instead of 10 rows (i.e., 10 x 1), to reduce 
   the number of vbroadcasts to access A

   this should achieve close-to-peak performance
   on Haswell (32flops/clock)
 */

#if __AVX512F__
enum {
  dM = 6,
  dN = 2
};
#elif __AVX__
enum {
  dM = 5, 
  dN = 2
};
#else
#error "define __AVX512F__ or __AVX__"
#endif

long gemm(matrix& A, matrix& B, matrix& C) {
  long M = C.nR, N = C.nC, K = A.nC;
  assert(M % dM == 0);
  assert(N % (dN * L) == 0);
  for (long i = 0; i < M; i += dM) {
    for (long j = 0; j < N; j += dN * L) {
      floatv c[dM][dN];
      for (long di = 0; di < dM; di++) {
	for (long dj = 0; dj < dN; dj++) {
	  c[di][dj] = C.v(i + di,j + dj * L);
	}
      }
      asm volatile("# begin");
      for (long k = 0; k < K; k++) {
	for (long di = 0; di < dM; di++) {
	  for (long dj = 0; dj < dN; dj++) {
	    c[di][dj] += A(i + di,k) * B.v(k,j + dj * L);
	  }
	}
      }
      asm volatile("# end");
      for (long di = 0; di < dM; di++) {
	for (long dj = 0; dj < dN; dj++) {
	  C.v(i + di,j + dj * L) = c[di][dj];
	}
      }
    }
  }
  return M * (N / L) * K;
}

