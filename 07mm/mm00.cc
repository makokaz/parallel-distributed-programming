#include "mm.h"

long gemm(matrix& A, matrix& B, matrix& C) {
  long M = C.nR, N = C.nC, K = A.nC;
  for (long i = 0; i < M; i++) {
    for (long j = 0; j < N; j++) {
      for (long k = 0; k < K; k++) {
	C(i,j) += A(i,k) * B(k,j);
      }
    }
  }
  return M * N * K;
}
