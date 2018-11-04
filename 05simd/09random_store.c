#include <assert.h>

void loop_random_store(float a, float * restrict x, long * idx, float b,
                       float * restrict y, long n) {
  /* tell the compiler x and y are 64 bytes-aligned (a multiple of 64) */
  x = __builtin_assume_aligned(x, 64);
  y = __builtin_assume_aligned(y, 64);
  /* tell the compiler n is a multiple of 16 */
  n = (n / 16) * 16;
  asm volatile("# loop begins");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    y[idx[i]] += a * x[i] + b;
  }
  asm volatile("# loop ends");
}
