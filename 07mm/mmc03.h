/* 
 * mmc03.h
 */
#include "mmc.h"

/* 
 * vectorize; 
 * concurrently update several (dM = nV/dN) rows and severl (dN) columns of C
 */

template<idx_t M,idx_t N,idx_t K,
  idx_t lda,idx_t ldb,idx_t ldc,
  idx_t bM,idx_t bN>
long gemm(matrix_c<M,K,lda>& A, matrix_c<K,N,ldb>& B, matrix_c<M,N,ldc>& C) {
  assert(M % bM == 0);
  assert(bN % L == 0);
  assert(N % bN == 0);
  for (idx_t i = 0; i < M; i += bM) {
    for (idx_t j = 0; j < N; j += bN) {
      asm volatile("# loop begins (%0,%1)x(%1,%2)" :: "i" (bM), "i" (K), "i" (bN));
      for (idx_t k = 0; k < K; k++) {
	for (idx_t ii = i; ii < i + bM; ii++) {
	  for (idx_t jj = j; jj < j + bN; jj += L) {
            C.v(ii,jj) += A(ii,k) * B.v(k,jj);
          }
	}
      }
      asm volatile("# loop ends");
    }
  }
  return ((long)M / bM) * ((long)N / bN) * (long)K;
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
