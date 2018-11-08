/* 
 * mmc02.h
 */
#include "mmc.h"

/* 
 * vectorize; 
 * concurrently update several rows of C
 */

template<idx_t M,idx_t N,idx_t K,idx_t lda,idx_t ldb,idx_t ldc,idx_t nV,idx_t dN>
idx_t gemm(matrix_c<M,K,lda>& A, matrix_c<K,N,ldb>& B, matrix_c<M,N,ldc>& C) {
  const idx_t dM = nV;
  assert(M % dM == 0);
  assert(N % L == 0);
  for (idx_t i = 0; i < M; i += dM) {
    for (idx_t j = 0; j < N; j += L) {
      asm volatile("# loop begins");
      for (idx_t k = 0; k < K; k++) {
	for (idx_t ii = i; ii < i + dM; ii++) {
	  C.v(ii,j) += A(ii,k) * B.v(k,j);
	}
      }
      asm volatile("# loop ends");
    }
  }
  return (M / dM) * (N / L) * K;
}


/* 

                              L columns (vector width)
                                     +--------+
                                     |   B    |
                                     |        |
                                     |bbbbbbbb|
                                     |        |
                                     |        |
                                     +--------+

        +-------- ... --------+      +--------+
        |    a                |    0 |cccccccc|
        |    a                |    1 |cccccccc|
dM rows |    a     A          |    2 |cccccccc|
        |    a                |    . |cccccccc|      
        |    a                |    . |cccccccc|      
        |    a                |    . |cccccccc|      
        +-------- ... --------+      +--------+      


 */
