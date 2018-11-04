#include <assert.h>

void loop_if(float a, float * restrict x, float b, float * restrict y, long n,
             long m) {
  /* tell the compiler x and y are 64 bytes-aligned (a multiple of 64) */
  x = __builtin_assume_aligned(x, 64);
  y = __builtin_assume_aligned(y, 64);
  /* tell the compiler n is a multiple of 16 */
  n = (n / 16) * 16;
  asm volatile("# loop begins");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    y[i] = x[i];
    for (long j = 0; j < m; j++) {
      y[i] = a * y[i] + b;
    }
  }
  asm volatile("# loop ends");
}
