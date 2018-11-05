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
enum { valign = sizeof(float) };
typedef float floatv __attribute__((vector_size(vwidth),aligned(valign)));
enum { L = sizeof(floatv) / sizeof(float) };
//N_SIMD_LANES = sizeof(floatv) / sizeof(float),

#ifdef __cplusplus
extern "C" {
#endif
  static inline void * alloc64(size_t sz) {
    void * a = 0;
    int r = posix_memalign(&a, 64, sz);
    assert(r == 0);
    return a;
  }
#ifdef __cplusplus
};
#endif

#ifdef __cplusplus

struct matrix {
  long nR;			// number of rows
  long nC;			// number of columns
  long ld;			// leading dimension
  float * a;			// pointer to data
  matrix() { }
  matrix(long nR_, long nC_, long ld_, float * a_=0) {
    init(nR_, nC_, ld_, a_);
  }
  void init(long nR_, long nC_, long ld_, float * a_) {
    nR = nR_;
    nC = nC_;
    ld = ld_;
    if (a_) {
      a = a_;
    } else {
      a = (float*)alloc64(sizeof(float) * nR * ld);
    }
  }
  // return A(i,j)
  float& operator() (long i, long j) {
    return a[i * ld + j];
  }
  // return A(i,j:j+8)
  floatv& v(long i, long j) {
    return *((floatv*)&a[i * ld + j]);
  }
  void split_r(matrix c[2], long align) {
    long nr = (nR / 2) - ((nR / 2) & (align - 1));
    c[0].init(nr,      nC, ld, a);
    c[1].init(nR - nr, nC, ld, &a[nr * ld]);
  }
  void split_c(matrix c[2], long align) {
    long nc = (nC / 2) - ((nC / 2) & (align - 1));
    c[0].init(nR, nc,      ld,  a);
    c[1].init(nR, nC - nc, ld, &a[nc]);
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

long gemm(matrix& A, matrix& B, matrix& C);

#endif

/* Intel architecture analzer */

#if IACA
#define ca_begin() do { asm volatile(".byte 0x0F, 0x0B"); asm volatile("movl $111,%ebx"); asm volatile(".byte 0x64, 0x67, 0x90"); } while(0)
#define ca_end()   do { asm volatile("movl $222,%ebx"); asm volatile(".byte 0x64, 0x67, 0x90"); asm volatile(".byte 0x0F, 0x0B"); } while(0)

/*
.byte 0x0F, 0x0B
movl $111,%ebx
.byte 0x64, 0x67, 0x90

movl $222,%ebx
.byte 0x64, 0x67, 0x90
.byte 0x0F, 0x0B

*/

#else
#define ca_begin()
#define ca_end()
#endif

