/* 
 * mmc00.h
 */
#include "mmc.h"

template<int M,int N,int K,int lda,int ldb,int ldc,int dM,int dN>
long gemm(matrix_c<M,K,lda>& A, matrix_c<K,N,ldb>& B, matrix_c<M,N,ldc>& C) {
  for (long i = 0; i < M; i++) {
    for (long j = 0; j < N; j++) {
      asm volatile("# loop begins");
      for (long k = 0; k < K; k++) {
	C(i,j) += A(i,k) * B(k,j);
      }
      asm volatile("# loop ends");
    }
  }
  return M * N * K;
}
