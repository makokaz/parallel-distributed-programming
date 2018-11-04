#include <assert.h>

#pragma omp declare simd uniform(a, x, b, y) linear(i:1) notinbranch
void f(float a, float * restrict x, float b, float * restrict y, long i);

void loop_fun(float a, float * restrict x, float b, float * restrict y, long n) {
  /* tell the compiler x and y are 64 bytes-aligned (a multiple of 64) */
  x = __builtin_assume_aligned(x, 64);
  y = __builtin_assume_aligned(y, 64);
  /* tell the compiler n is a multiple of 16 */
  n = (n / 16) * 16;
  asm volatile("# loop begins");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    f(a, x, b, y, i);
  }
  asm volatile("# loop ends");
}
