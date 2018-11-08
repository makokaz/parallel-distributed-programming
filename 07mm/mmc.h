/* 
 * mm.h
 */

#pragma once

#include <assert.h>
#include <stdlib.h>
typedef float real;
typedef long idx_t;

#if __AVX512F__
enum { vwidth = 64 };
#elif __AVX__
enum { vwidth = 32 };
#else
#error "__AVX512F__ or __AVX__ must be defined"
#endif
enum {
  //valign = sizeof(real),
  valign = vwidth
};
typedef real realv __attribute__((vector_size(vwidth),aligned(valign)));
enum { L = sizeof(realv) / sizeof(real) };

#define CHECK_IDX 0

/* matrix with constant size and leading dimension */
template<idx_t nR,idx_t nC,idx_t ld>
struct matrix_c {
  real a[nR][ld] __attribute__((aligned(vwidth)));
  matrix_c() { }
  // return A(i,j)
  real& operator() (idx_t i, idx_t j) {
#if CHECK_IDX
    assert(i < nR);
    assert(j < nC);
    assert(i >= 0);
    assert(j >= 0);
#endif
    return a[i][j];
  }
  // return A(i,j:j+8)
  realv& v(idx_t i, idx_t j) {
#if CHECK_IDX
    assert(i < nR);
    assert(j < nC);
    assert(i >= 0);
    assert(j >= 0);
#endif
    return *((realv*)&a[i][j]);
  }
  void rand_init(unsigned short rg[3]) {
    for (idx_t i = 0; i < nR; i++) {
      for (idx_t j = 0; j < nC; j++) {
	(*this)(i,j) = erand48(rg);
      }
    }
  }
  void const_init(real c) {
    for (idx_t i = 0; i < nR; i++) {
      for (idx_t j = 0; j < nC; j++) {
	(*this)(i,j) = c;
      }
    }
  }
  void zero() {
    const_init(0.0);
  }
};





