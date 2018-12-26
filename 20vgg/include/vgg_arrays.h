/**
   @file vgg_arrays.h --- vectors and arrays
 */

#pragma once

#include <stdio.h>
#include "vgg_util.h"

static void range_chk_(idx_t a, idx_t x, idx_t b,
                       const char * a_str, const char * x_str,
                       const char * b_str, 
                       const char * file, int line) {
  if (!(a <= x)) {
    fprintf(stderr,
            "error:%s:%d: index check [%s <= %s < %s] failed"
            " (%s = %ld)\n",
            file, line, a_str, x_str, b_str, x_str, (long)x);
    bail();
  }
  if (!(x < b)) {
    fprintf(stderr,
            "error:%s:%d: index check [%s <= %s < %s] failed"
            " (%s = %ld)\n",
            file, line, a_str, x_str, b_str, x_str, (long)x);
    bail();
  }
}

/* compile it with -DARRAY_INDEX_CHECK=1 to check array index overflow */
#if ARRAY_INDEX_CHECK
#define range_chk(a, x, b) range_chk_(a, x, b, #a, #x, #b, __FILE__, __LINE__)
#else
#define range_chk(a, x, b) 
#endif

/**
   @brief vec
 */
template<idx_t N>
struct vec {
  idx_t n;
  real w[N];
  real& operator()(idx_t i) {
    range_chk(0, i, n);
    return w[i];
  }
  void init(idx_t n_) {
    n = n_;
    assert(n <= N);
  }
  void init_const(idx_t n_, real x) {
    init(n_);
    vec<N>& v = *this;
    for (idx_t i = 0; i < n; i++) {
      v(i) = x;
    }
  }
  void init_uniform(idx_t n_, rnd_gen_t& rg, real p, real q) {
    init(n_);
    vec<N>& v = *this;
    for (idx_t i = 0; i < n; i++) {
      v(i) = rg.rand(p, q);
    }
  }
  void init_normal(idx_t n_, rnd_gen_t& rg, real mu, real sigma) {
    init(n_);
    vec<N>& v = *this;
    for (idx_t i = 0; i < n; i++) {
      v(i) = rg.rand_normal(mu, sigma);
    }
  }
  void update(real eta, vec<N>& dw) {
    // a += eta * b
    vec<N>& a = *this;
    for (idx_t i = 0; i < n; i++) {
      a(i) += eta * dw(i);
    }
  }
  real sum() {
    vec<N>& v = *this;
    real s = 0.0;
    for (idx_t i = 0; i < n; i++) {
      s += v(i);
    }
    return s;
  }
};

/**
   @brief vec
 */
template<idx_t N>
struct ivec {
  idx_t n;
  idx_t w[N];
  idx_t& operator()(idx_t i) {
    range_chk(0, i, n);
    return w[i];
  }
  void init(idx_t n_) {
    n = n_;
    assert(n <= N);
  }
  void init_const(idx_t n_, idx_t x) {
    init(n_);
    ivec<N>& v = *this;
    for (idx_t i = 0; i < n; i++) {
      v(i) = x;
    }
  }
  void init_uniform(idx_t n_, rnd_gen_t& rg, idx_t a, idx_t b) {
    init(n_);
    ivec<N>& v = *this;
    for (idx_t i = 0; i < n; i++) {
      v(i) = rg.randi(a, b);
    }
  }
};

/**
   @brief matrix (2D array)
 */
template<idx_t M,idx_t N>
struct array2 {
  idx_t m;
  real w[M][N];
  real& operator()(idx_t i, idx_t j) {
    range_chk(0, i, m);
    range_chk(0, j, N);
    return w[i][j];
  }
  void init(idx_t m_) {
    m = m_;
    assert(m <= M);
  }
  void init_const(idx_t m_, real x) {
    init(m_);
    array2<M,N>& a = *this;
    for (idx_t i = 0; i < m; i++) {
      for (idx_t j = 0; j < N; j++) {
        a(i,j) = x;
      }
    }
  }
  void init_uniform(idx_t m_, rnd_gen_t& rg, real p, real q) {
    init(m_);
    array2<M,N>& a = *this;
    for (idx_t i = 0; i < m; i++) {
      for (idx_t j = 0; j < N; j++) {
        a(i,j) = rg.rand(p, q);
      }
    }
  }
  void init_normal(idx_t m_, rnd_gen_t& rg, real mu, real sigma) {
    init(m_);
    array2<M,N>& a = *this;
    for (idx_t i = 0; i < m; i++) {
      for (idx_t j = 0; j < N; j++) {
        a(i,j) = rg.rand_normal(mu, sigma);
      }
    }
  }
  void update(real eta, array2<M,N>& dw) {
    // a += eta * b
    array2<M,N>& a = *this;
    assert(m == dw.m);
    for (idx_t i = 0; i < m; i++) {
      for (idx_t j = 0; j < N; j++) {
        a(i,j) += eta * dw(i,j);
      }
    }
  }
};

/**
   @brief maxBxCxHxW tensor (4D array)
 */
template<idx_t maxB,idx_t C,idx_t H,idx_t W>
struct array4 {
  idx_t B;
  real w[maxB][C][H][W];
  real& operator()(idx_t b, idx_t c, idx_t i, idx_t j) {
    range_chk(0, b, B);
    range_chk(0, c, C);
    range_chk(0, i, H);
    range_chk(0, j, W);
    return w[b][c][i][j];
  }
  void init(idx_t B_) {
    B = B_;
    assert(B <= maxB);
  }
  void init_const(idx_t B_, real x) {
    init(B_);
    array4<maxB,C,H,W>& a = *this;
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            a(b,c,i,j) = x;
          }
        }
      }
    }
  }
  void init_uniform(idx_t B_, rnd_gen_t& rg, real p, real q) {
    init(B_);
    array4<maxB,C,H,W>& a = *this;
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            a(b,c,i,j) = rg.rand(p, q);
          }
        }
      }
    }
  }
  void init_normal(idx_t B_, rnd_gen_t& rg, real mu, real sigma) {
    init(B_);
    array4<maxB,C,H,W>& a = *this;
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            a(b,c,i,j) = rg.rand_normal(mu, sigma);
          }
        }
      }
    }
  }
  void update(real eta, array4<maxB,C,H,W>& dw) {
    // a += eta * b
    array4<maxB,C,H,W>& a = *this;
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            a(b,c,i,j) += eta * dw(b,c,i,j);
          }
        }
      }
    }
  }
};

/**
   @brief 4D array for convolution
 */
template<idx_t OC,idx_t IC,idx_t H,idx_t W>
struct warray4 {
  real w[OC][IC][2*H+1][2*W+1];
  real& operator()(idx_t oc, idx_t ic, idx_t i, idx_t j) {
    range_chk(0, oc, OC);
    range_chk(0, ic, IC);
    range_chk(-H, i, H+1);
    range_chk(-W, j, W+1);
    return w[oc][ic][i+H][j+W];
  }
  void init_const(real x) {
    warray4<OC,IC,H,W>& a = *this;
    for (idx_t oc = 0; oc < OC; oc++) {
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = -H; i <= H; i++) {
          for (idx_t j = -W; j <= W; j++) {
            a(oc,ic,i,j) = x;
          }
        }
      }
    }
  }
  void init_uniform(rnd_gen_t& rg, real p, real q) {
    warray4<OC,IC,H,W>& a = *this;
    for (idx_t oc = 0; oc < OC; oc++) {
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = -H; i <= H; i++) {
          for (idx_t j = -W; j <= W; j++) {
            a(oc,ic,i,j) = rg.rand(p, q);
          }
        }
      }
    }
  }
  void init_normal(rnd_gen_t& rg, real mu, real sigma) {
    warray4<OC,IC,H,W>& a = *this;
    for (idx_t oc = 0; oc < OC; oc++) {
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = -H; i <= H; i++) {
          for (idx_t j = -W; j <= W; j++) {
            a(oc,ic,i,j) = rg.rand_normal(mu, sigma);
          }
        }
      }
    }
  }
  void update(real eta, warray4<OC,IC,H,W>& dw) {
    // a += eta * b
    warray4<OC,IC,H,W>& a = *this;
    for (idx_t oc = 0; oc < OC; oc++) {
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = -H; i <= H; i++) {
          for (idx_t j = -W; j <= W; j++) {
            a(oc,ic,i,j) += eta * dw(oc,ic,i,j);
          }
        }
      }
    }
  }
};

int vgg_arrays_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const int N = 10;
  const int n = 8;
  vec<N> v;
  ivec<N> iv;
  array2<N,N> a2;
  array4<N,N,N,N> a4;
  warray4<N,N,N,N> w4;
  v.init_const(n, 1);
  iv.init_const(n, 2);
  a2.init_const(n, 3);
  a4.init_const(n, 4);
  w4.init_const(5);
  v(n-1) = opt.iters;
  iv(n-1) = 6;
  a2(n-1,n-1) = 7;
  a4(n-1,n-1,n-1,n-1) = 8;
  w4(n-1,n-1,n-1,n-1) = 9;
  return 0;
}

void vgg_arrays_use_unused_functions() {
  (void)range_chk_;
}
