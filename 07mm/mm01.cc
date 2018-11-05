#include "mm.h"

/* just vectorize along j axis */
long gemm(matrix& A, matrix& B, matrix& C) {
  long M = C.nR, N = C.nC, K = A.nC;
  assert(N % L == 0);
  for (long i = 0; i < M; i++) {
    for (long j = 0; j < N; j += L) {
      for (long k = 0; k < K; k++) {
	C.v(i,j) += A(i,k) * B.v(k,j);
      }
    }
  }
  return M * (N / L) * K;
}

