/* 
 * mm.h
 */

#pragma once

#include <assert.h>
#include <stdlib.h>
#if __AVX512F__
enum { vwidth = 64 };
#elif __AVX__
enum { vwidth = 32 };
#else
#error "__AVX512F__ or __AVX__ must be defined"
#endif
enum {
  //valign = sizeof(float),
  valign = vwidth
};
typedef float floatv __attribute__((vector_size(vwidth),aligned(valign)));
enum { L = sizeof(floatv) / sizeof(float) };

#define CHECK_IDX 0

/* matrix with constant size and leading dimension */
template<int nR,int nC,int ld>
struct matrix_c {
  float a[nR][ld] __attribute__((aligned(vwidth)));
  matrix_c() { }
  // return A(i,j)
  float& operator() (long i, long j) {
#if CHECK_IDX
    assert(i < nR);
    assert(j < nC);
    assert(i >= 0);
    assert(j >= 0);
#endif
    return a[i][j];
  }
  // return A(i,j:j+8)
  floatv& v(long i, long j) {
#if CHECK_IDX
    assert(i < nR);
    assert(j < nC);
    assert(i >= 0);
    assert(j >= 0);
#endif
    return *((floatv*)&a[i][j]);
  }
  void rand_init(unsigned short rg[3]) {
    for (long i = 0; i < nR; i++) {
      for (long j = 0; j < nC; j++) {
	(*this)(i,j) = erand48(rg);
      }
    }
  }
  void const_init(float c) {
    for (long i = 0; i < nR; i++) {
      for (long j = 0; j < nC; j++) {
	(*this)(i,j) = c;
      }
    }
  }
  void zero() {
    const_init(0.0);
  }
};





