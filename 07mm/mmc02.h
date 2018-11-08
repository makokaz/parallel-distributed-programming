/* 
 * mmc02.h
 */
#include "mmc.h"

/* 
 * vectorize; 
 * concurrently update several rows of C
 */

template<int M,int N,int K,int lda,int ldb,int ldc,int nV,int dN>
long gemm(matrix_c<M,K,lda>& A, matrix_c<K,N,ldb>& B, matrix_c<M,N,ldc>& C) {
  const int dM = nV;
  assert(M % dM == 0);
  assert(N % L == 0);
  for (long i = 0; i < M; i += dM) {
    for (long j = 0; j < N; j += L) {
      asm volatile("# loop begins");
      for (long k = 0; k < K; k++) {
	for (long ii = i; ii < i + dM; ii++) {
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
