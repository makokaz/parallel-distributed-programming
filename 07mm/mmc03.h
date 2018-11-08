/* 
 * mmc03.h
 */
#include "mmc.h"

/* 
 * vectorize; 
 * concurrently update several (dM = nV/dN) rows and severl (dN) columns of C
 */

template<int M,int N,int K,int lda,int ldb,int ldc,int nV,int dN>
long gemm(matrix_c<M,K,lda>& A, matrix_c<K,N,ldb>& B, matrix_c<M,N,ldc>& C) {
  assert(nV % dN == 0);
  const int dM = nV / dN;
  assert(M % dM == 0);
  assert(N % (dN * L) == 0);
  for (long i = 0; i < M; i += dM) {
    for (long j = 0; j < N; j += dN * L) {
      asm volatile("# loop begins");
      for (long k = 0; k < K; k++) {
	for (long di = 0; di < dM; di++) {
	  for (long dj = 0; dj < dN * L; dj += L) {
            C.v(i+di,j+dj) += A(i+di,k) * B.v(k,j+dj);
          }
	}
      }
      asm volatile("# loop ends");
    }
  }
  return (M / dM) * (N / (dN * L)) * K;
}

/* 

                                  dN * L columns (L = vector width)
                                     +--------+--------+
                                     |   B    |        |
                                     |        |        |
                                     |bbbbbbbb|bbbbbbbb|
                                     |        |        |
                                     |        |        |
                                     +--------+--------+

        +-------- ... --------+      +--------+--------+
        |    a                |    0 |cccccccc|cccccccc|
        |    a                |    1 |cccccccc|cccccccc| (dM * dN) vector 
dM rows |    a     A          |    2 |cccccccc|cccccccc| elements updated
        |    a                |    . |cccccccc|cccccccc|
        +-------- ... --------+      +--------+--------+

 */
