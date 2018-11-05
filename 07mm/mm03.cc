/* 
 * mm4.cc
 */

#include "mm.h"

#if __AVX512F__
enum {
  dM = 8
};
#elif __AVX__
enum {
  dM = 10
};
#else
#error "define __AVX512F__ or __AVX__"
#endif

/* similar to mm_3.h, but helps the compiler allocate
   registers to updated rows of C.
 */

long gemm(matrix& A, matrix& B, matrix& C) {
  long M = C.nR, N = C.nC, K = A.nC;
  assert(M % dM == 0);
  assert(N % L == 0);
  for (long i = 0; i < M; i += dM) {
    for (long j = 0; j < N; j += 8) {
      floatv c[dM];
      for (long di = 0; di < dM; di++) {
	c[di] = C.v(i + di,j);
      }
      asm volatile("# begin");
      for (long k = 0; k < K; k++) {
	for (long di = 0; di < dM; di++) {
	  c[di] += A(i + di,k) * B.v(k,j);
	}
      }
      asm volatile("# end");
      for (long di = 0; di < dM; di++) {
	C.v(i + di,j) = c[di];
      }
    }
  }
  return M * (N / L) * K;
}

