/* 
 * mm3.h
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

/* 
 * vectorize; 
 * concurrently update several rows of C
 */

long gemm(matrix& A, matrix& B, matrix& C) {
  long M = C.nR, N = C.nC, K = A.nC;
  assert(M % dM == 0);
  assert(N % L == 0);
  for (long i = 0; i < M; i += dM) {
    for (long j = 0; j < N; j += L) {
      asm volatile("# begin");
      for (long k = 0; k < K; k++) {
	for (long ii = i; ii < i + dM; ii++) {
	  C.v(ii,j) += A(ii,k) * B.v(k,j);
	}
      }
      asm volatile("# end");
    }
  }
  return M * (N / L) * K;
}

/* 
                                                   01234567
                                                  +--------+
                                                  |        |
      01234567                  K                 |bbbbbbbb|
     +--------+      +-------- ... --------+      |        |
   0 |        |      |    a                |      .        .
   1 |        |      |    a                |      .        .
   2 |        |  +=  |    a                | *  K .        .
   3 |        |      |    a                |      .        .
   4 |        |      |    a                |      .        .
   5 |        |      |    a                |      .        .
   6 |        |      |    a                |      .        .
   7 |        |      |    a                |      .        .
   8 |        |      |    a                |      .        .
   9 |        |      |    a                |      |        |
     +--------+      +-------- ... --------+      |        |
                                                  |        |
                                                  +--------+

 */
