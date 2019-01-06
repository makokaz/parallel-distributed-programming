/**
   @file vgg_arrays.h --- vectors and arrays
 */

#pragma once

#include <stdio.h>
#include "vgg_util.h"

__device__ __host__ 
static void range_chk_(idx_t a, idx_t x, idx_t b,
                       const char * a_str, const char * x_str,
                       const char * b_str, 
                       const char * file, int line) {
#if ! __CUDA_ARCH__
  if (!(a <= x)) {
    fprintf(stderr,
            "error:%s:%d: index check [%s <= %s < %s] failed"
            " (%s = %ld)\n",
            file, line, a_str, x_str, b_str, x_str, (long)x);
  }
#endif
  assert(a <= x);
#if ! __CUDA_ARCH__
  if (!(x < b)) {
    fprintf(stderr,
            "error:%s:%d: index check [%s <= %s < %s] failed"
            " (%s = %ld)\n",
            file, line, a_str, x_str, b_str, x_str, (long)x);
  }
#endif
  assert(x < b);
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
#if __NVCC__
  vec<N> * dev;
#endif
  idx_t n;
  real w[N];
  __device__ __host__ 
  real& operator()(idx_t i) {
    range_chk(0, i, n);
    return w[i];
  }
  __device__ __host__ 
  void set_n(idx_t n_) {
    n = n_;
    assert(n <= N);
  }
  __device__ __host__ 
  void init_const(idx_t n_, real c) {
    set_n(n_);
    vec<N>& x = *this;
    for (idx_t i = 0; i < n; i++) {
      x(i) = c;
    }
  }
  void init_uniform(idx_t n_, rnd_gen_t& rg, real p, real q) {
    set_n(n_);
    vec<N>& x = *this;
    for (idx_t i = 0; i < n; i++) {
      x(i) = rg.rand(p, q);
    }
  }
  void init_single(idx_t n_, rnd_gen_t& rg, real p, real q) {
    vec<N>& x = *this;
    x.init_const(n_, 0);
    idx_t i = rg.randi(0, n);
    x(i) = rg.rand(p, q);
  }
  void init_normal(idx_t n_, rnd_gen_t& rg, real mu, real sigma) {
    set_n(n_);
    vec<N>& x = *this;
    for (idx_t i = 0; i < n; i++) {
      x(i) = rg.rand_normal(mu, sigma);
    }
  }
  __device__ __host__
  void update(real eta, vec<N>& dx) {
    // this += eta * dv
    vec<N>& x = *this;
    assert(x.n == dx.n);
    for (idx_t i = 0; i < n; i++) {
      x(i) += eta * dx(i);
    }
  }
  real sum() {
    vec<N>& x = *this;
    real s = 0.0;
    for (idx_t i = 0; i < n; i++) {
      s += x(i);
    }
    return s;
  }
  real dot(vec<N>& y) {
    vec<N>& x = *this;
    assert(x.n == y.n);
    real s = 0.0;
    for (idx_t i = 0; i < n; i++) {
      s += x(i) * y(i);
    }
    return s;
  }
  real diff(vec<N>& y) {
    vec<N>& x = *this;
    assert(x.n == y.n);
    real s = 0.0;
    for (idx_t i = 0; i < n; i++) {
      real d = x(i) - y(i);
      s += d * d;
    }
    return s;
  }
  void set_dev(vec<N> * dev_) {
#if __NVCC__
    dev = dev_;
#endif
  }
  void make_dev(int gpu) {
#if __NVCC__
    if (gpu) {
      dev = (vec<N>*)dev_malloc(sizeof(*this));
    } else {
      dev = 0;
    }
#else
    (void)gpu;
#endif
  }
  void del_dev() {
#if __NVCC__
    if (dev) {
      dev_free(dev);
      dev = 0;
    }
#endif
  }
  void to_dev() {
#if __NVCC__
    if (dev) {
      ::to_dev(dev, this, sizeof(*this));
    }
#endif
  }
  void to_host() {
#if __NVCC__
    if (dev) {
      vec<N> * dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
};

/**
   @brief vec
 */
template<idx_t N>
struct ivec {
#if __NVCC__
  ivec<N> * dev;
#endif
  idx_t n;
  idx_t w[N];
  __device__ __host__ 
  idx_t& operator()(idx_t i) {
    range_chk(0, i, n);
    return w[i];
  }
  __device__ __host__ 
  void set_n(idx_t n_) {
    n = n_;
    assert(n <= N);
  }
  void init_const(idx_t n_, idx_t c) {
    set_n(n_);
    ivec<N>& x = *this;
    for (idx_t i = 0; i < n; i++) {
      x(i) = c;
    }
  }
  void init_uniform(idx_t n_, rnd_gen_t& rg, idx_t p, idx_t q) {
    set_n(n_);
    ivec<N>& x = *this;
    for (idx_t i = 0; i < n; i++) {
      x(i) = rg.randi(p, q);
    }
  }
  void set_dev(ivec<N> * dev_) {
#if __NVCC__
    dev = dev_;
#endif
  }
  void make_dev(int gpu) {
#if __NVCC__
    if (gpu) {
      dev = (ivec<N>*)dev_malloc(sizeof(*this));
    } else {
      dev = 0;
    }
#else
    (void)gpu;
#endif
  }
  void del_dev() {
#if __NVCC__
    if (dev) {
      dev_free(dev);
      dev = 0;
    }
#endif
  }
  void to_dev() {
#if __NVCC__
    if (dev) {
      ::to_dev(dev, this, sizeof(*this));
    }
#endif
  }
  void to_host() {
#if __NVCC__
    if (dev) {
      ivec<N> * dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
};

/**
   @brief matrix (2D array)
 */
template<idx_t M,idx_t N>
struct array2 {
#if __NVCC__
  array2<M,N> * dev;
#endif
  idx_t m;
  real w[M][N];
  __device__ __host__ 
  real& operator()(idx_t i, idx_t j) {
    range_chk(0, i, m);
    range_chk(0, j, N);
    return w[i][j];
  }
  __device__ __host__ 
  void set_n_rows(idx_t m_) {
    m = m_;
    assert(m <= M);
  }
  void init_const(idx_t m_, real x) {
    set_n_rows(m_);
    array2<M,N>& a = *this;
    for (idx_t i = 0; i < m; i++) {
      for (idx_t j = 0; j < N; j++) {
        a(i,j) = x;
      }
    }
  }
  void init_uniform(idx_t m_, rnd_gen_t& rg, real p, real q) {
    set_n_rows(m_);
    array2<M,N>& a = *this;
    for (idx_t i = 0; i < m; i++) {
      for (idx_t j = 0; j < N; j++) {
        a(i,j) = rg.rand(p, q);
      }
    }
  }
  void init_normal(idx_t m_, rnd_gen_t& rg, real mu, real sigma) {
    set_n_rows(m_);
    array2<M,N>& a = *this;
    for (idx_t i = 0; i < m; i++) {
      for (idx_t j = 0; j < N; j++) {
        a(i,j) = rg.rand_normal(mu, sigma);
      }
    }
  }
  __device__ __host__
  void update(real eta, array2<M,N>& da) {
    // a += eta * da
    array2<M,N>& a = *this;
    assert(a.m == da.m);
    for (idx_t i = 0; i < m; i++) {
      for (idx_t j = 0; j < N; j++) {
        a(i,j) += eta * da(i,j);
      }
    }
  }
  real dot(array2<M,N>& b) {
    array2<M,N>& a = *this;
    assert(a.m == b.m);
    real s0 = 0.0;
    for (idx_t i = 0; i < m; i++) {
      real s1 = 0.0;
      for (idx_t j = 0; j < N; j++) {
        s1 += a(i,j) * b(i,j);
      }
      s0 += s1;
    }
    return s0;
  }
  real diff(array2<M,N>& b) {
    array2<M,N>& a = *this;
    assert(a.m == b.m);
    real s0 = 0.0;
    for (idx_t i = 0; i < m; i++) {
      real s1 = 0.0;
      for (idx_t j = 0; j < N; j++) {
        real d = a(i,j) - b(i,j);
        s1 += d * d;
      }
      s0 += s1;
    }
    return s0;
  }
  void set_dev(array2<M,N> * dev_) {
#if __NVCC__
    dev = dev_;
#endif
  }
  void make_dev(int gpu) {
#if __NVCC__
    if (gpu) {
      dev = (array2<M,N>*)dev_malloc(sizeof(*this));
    } else {
      dev = 0;
    }
#else
    (void)gpu;
#endif
  }
  void del_dev() {
#if __NVCC__
    if (dev) {
      dev_free(dev);
      dev = 0;
    }
#endif
  }
  void to_dev() {
#if __NVCC__
    if (dev) {
      ::to_dev(dev, this, sizeof(*this));
    }
#endif
  }
  void to_host() {
#if __NVCC__
    if (dev) {
      array2<M,N> * dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
};

/**
   @brief maxBxCxHxW tensor (4D array)
 */
template<idx_t maxB,idx_t C,idx_t H,idx_t W>
struct array4 {
#if __NVCC__
  array4<maxB,C,H,W> * dev;
#endif
  idx_t B;
  real w[maxB][C][H][W];
  __device__ __host__ 
  real& operator()(idx_t b, idx_t c, idx_t i, idx_t j) {
    range_chk(0, b, B);
    range_chk(0, c, C);
    range_chk(0, i, H);
    range_chk(0, j, W);
    return w[b][c][i][j];
  }
  __device__ __host__ 
  void set_n_rows(idx_t B_) {
    B = B_;
    assert(B <= maxB);
  }
  void init_const(idx_t B_, real x) {
    set_n_rows(B_);
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
  void init_single(idx_t B_, rnd_gen_t& rg, real p, real q) {
    array4<maxB,C,H,W>& a = *this;
    a.init_const(B_, 0);
    idx_t b = rg.randi(0, B);
    idx_t c = rg.randi(0, C);
    idx_t i = rg.randi(0, H);
    idx_t j = rg.randi(0, W);
    a(b,c,i,j) = rg.rand(p, q);
  }
  void init_uniform(idx_t B_, rnd_gen_t& rg, real p, real q) {
    set_n_rows(B_);
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
    set_n_rows(B_);
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
  __device__ __host__
  void update(real eta, array4<maxB,C,H,W>& da) {
    array4<maxB,C,H,W>& a = *this;
    assert(a.B == da.B);
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            a(b,c,i,j) += eta * da(b,c,i,j);
          }
        }
      }
    }
  }
  real dot(array4<maxB,C,H,W>& a_) {
    array4<maxB,C,H,W>& a = *this;
    assert(a.B == a_.B);
    real s0 = 0.0;
    for (idx_t b = 0; b < B; b++) {
      real s1 = 0.0;
      for (idx_t c = 0; c < C; c++) {
        real s2 = 0.0;
        for (idx_t i = 0; i < H; i++) {
          real s3 = 0.0;
          for (idx_t j = 0; j < W; j++) {
            s3 += a(b,c,i,j) * a_(b,c,i,j);
          }
          s2 += s3;
        }
        s1 += s2;
      }
      s0 += s1;
    }
    return s0;
  }
  real diff(array4<maxB,C,H,W>& a_) {
    array4<maxB,C,H,W>& a = *this;
    assert(a.B == a_.B);
    real s0 = 0.0;
    for (idx_t b = 0; b < B; b++) {
      real s1 = 0.0;
      for (idx_t c = 0; c < C; c++) {
        real s2 = 0.0;
        for (idx_t i = 0; i < H; i++) {
          real s3 = 0.0;
          for (idx_t j = 0; j < W; j++) {
            real d = a(b,c,i,j) - a_(b,c,i,j);
            s3 += d * d;
          }
          s2 += s3;
        }
        s1 += s2;
      }
      s0 += s1;
    }
    return s0;
  }
  void set_dev(array4<maxB,C,H,W>* dev_) {
#if __NVCC__
    dev = dev_;
#endif
  }
  void make_dev(int gpu) {
#if __NVCC__
    if (gpu) {
      dev = (array4<maxB,C,H,W>*)dev_malloc(sizeof(*this));
    } else {
      dev = 0;
    }
#else
    (void)gpu;
#endif
  }
  void del_dev() {
#if __NVCC__
    if (dev) {
      dev_free(dev);
      dev = 0;
    }
#endif
  }
  void to_dev() {
#if __NVCC__
    if (dev) {
      ::to_dev(dev, this, sizeof(*this));
    }
#endif
  }
  void to_host() {
#if __NVCC__
    if (dev) {
      array4<maxB,C,H,W> * dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
};

/**
   @brief 4D array for convolution
 */
template<idx_t OC,idx_t IC,idx_t H,idx_t W>
struct warray4 {
#if __NVCC__
  warray4<OC,IC,H,W> * dev;
#endif
  real w[OC][IC][2*H+1][2*W+1];
  __device__ __host__ 
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
  __device__ __host__
  void update(real eta, warray4<OC,IC,H,W>& da) {
    // a += eta * da
    warray4<OC,IC,H,W>& a = *this;
    for (idx_t oc = 0; oc < OC; oc++) {
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = -H; i <= H; i++) {
          for (idx_t j = -W; j <= W; j++) {
            a(oc,ic,i,j) += eta * da(oc,ic,i,j);
          }
        }
      }
    }
  }
  real dot(warray4<OC,IC,H,W>& a_) {
    warray4<OC,IC,H,W>& a = *this;
    real s0 = 0.0;
    for (idx_t oc = 0; oc < OC; oc++) {
      real s1 = 0.0;
      for (idx_t ic = 0; ic < IC; ic++) {
        real s2 = 0.0;
        for (idx_t i = -H; i <= H; i++) {
          real s3 = 0.0;
          for (idx_t j = -W; j <= W; j++) {
            s3 += a(oc,ic,i,j) * a_(oc,ic,i,j);
          }
          s2 += s3;
        }
        s1 += s2;
      }
      s0 += s1;
    }
    return s0;
  }
  real diff(warray4<OC,IC,H,W>& a_) {
    warray4<OC,IC,H,W>& a = *this;
    real s0 = 0.0;
    for (idx_t oc = 0; oc < OC; oc++) {
      real s1 = 0.0;
      for (idx_t ic = 0; ic < IC; ic++) {
        real s2 = 0.0;
        for (idx_t i = -H; i <= H; i++) {
          real s3 = 0.0;
          for (idx_t j = -W; j <= W; j++) {
            real d = a(oc,ic,i,j) - a_(oc,ic,i,j);
            s3 += d * d;
          }
          s2 += s3;
        }
        s1 += s2;
      }
      s0 += s1;
    }
    return s0;
  }
  void set_dev(warray4<OC,IC,H,W>* dev_) {
#if __NVCC__
    dev = dev_;
#endif
  }
  void make_dev(int gpu) {
#if __NVCC__
    if (gpu) {
      dev = (warray4<OC,IC,H,W>*)dev_malloc(sizeof(*this));
    } else {
      dev = 0;
    }
#else
    (void)gpu;
#endif
  }
  void del_dev() {
#if __NVCC__
    if (dev) {
      dev_free(dev);
      dev = 0;
    }
#endif
  }
  void to_dev() {
#if __NVCC__
    if (dev) {
      ::to_dev(dev, this, sizeof(*this));
    }
#endif
  }
  void to_host() {
#if __NVCC__
    if (dev) {
      warray4<OC,IC,H,W> * dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
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
