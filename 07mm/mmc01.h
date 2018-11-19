/* 
 * mmc01.h
 */
#include "mmc.h"

/* vectorize along j axis */
template<idx_t M,idx_t N,idx_t K,
  idx_t lda,idx_t ldb,idx_t ldc,
  idx_t bM,idx_t bN>
long gemm(matrix_c<M,K,lda>& A, matrix_c<K,N,ldb>& B, matrix_c<M,N,ldc>& C) {
  assert(N % L == 0);
  for (idx_t i = 0; i < M; i++) {
    for (idx_t j = 0; j < N; j += L) {
      asm volatile("# loop begins (%0,%1)x(%1,%2)" :: "i" (1), "i" (K), "i" (L));
      for (idx_t k = 0; k < K; k++) {
	C.v(i,j) += A(i,k) * B.v(k,j);
      }
      asm volatile("# loop ends");
    }
  }
  return (long)M * ((long)N / L) * (long)K;
}

