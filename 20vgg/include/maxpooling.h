/**
   @file maxpooling.h
 */
#pragma once

#include "vgg_util.h"
#include "vgg_arrays.h"

#if __NVCC__
template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t S>
  struct MaxPooling2D;

template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t S>
  __global__ void forward_global(MaxPooling2D<maxB,C,H,W,S>* dev,
                                 array4<maxB,C,H,W>* x_dev) {
  dev->forward_dev(*x_dev);
}

template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t S>
  __global__ void backward_global(MaxPooling2D<maxB,C,H,W,S>* dev,
                                  array4<maxB,C,H/S,W/S>* gy_dev) {
  dev->backward_dev(*gy_dev);
}
#endif

template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t S>
struct MaxPooling2D {
#if __NVCC__
  MaxPooling2D<maxB,C,H,W,S>* dev;
#endif
  logger * lgr;
  cmdline_opt opt;
  array4<maxB,C,H/S,W/S> y;
  array4<maxB,C,H/S,W/S> max_idx;
  array4<maxB,C,H,W> gx;
  void init(cmdline_opt opt, logger * lgr) {
    this->opt = opt;
    this->lgr = lgr;
  }
  MaxPooling2D<maxB,C,H,W,S>* copy() {
    MaxPooling2D<maxB,C,H,W,S>* c = new MaxPooling2D<maxB,C,H,W,S>(*this);
    c->make_dev();
    return c;
  }
  void set_dev(MaxPooling2D<maxB,C,H,W,S>* dev) {
#if __NVCC__
    this->dev = dev;
    y.set_dev(dev ? &dev->y : 0);
    max_idx.set_dev(dev ? &dev->max_idx : 0);
    gx.set_dev(dev ? &dev->gx : 0);
#endif
  }
  void make_dev() {
#if __NVCC__
    if (opt.gpu_algo) {
      dev = (MaxPooling2D<maxB,C,H,W,S>*)dev_malloc(sizeof(*this));
    } else {
      dev = 0;
    }
    set_dev(dev);
#endif
  }
  void del_dev() {
#if __NVCC__
    if (opt.gpu_algo) {
      assert(dev);
      dev_free(dev);
      dev = 0;
    }
#endif
  }
  void to_dev() {
#if __NVCC__
    if (opt.gpu_algo) {
      assert(dev);
      ::to_dev(dev, this, sizeof(*this));
    }
#endif
  }
  void to_host() {
#if __NVCC__
    if (opt.gpu_algo) {
      assert(dev);
      MaxPooling2D<maxB,C,H,W,S>* dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
  __device__ __host__
  void forward_base(array4<maxB,C,H,W>& x) {
    const idx_t B = x.B;
    y.set_n_rows(B);
    max_idx.set_n_rows(B);
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H/S; i++) {
          for (idx_t j = 0; j < W/S; j++) {
            real s = x(b,c,S*i,S*j);
            idx_t idx = W * S * i  + S * j;
            for (idx_t i_ = S * i; i_ < S * (i + 1); i_++) {
              for (idx_t j_ = S * j; j_ < S * (j + 1); j_++) {
                if (s < x(b,c,i_,j_)) {
                  s = x(b,c,i_,j_);
                  idx = W * i_ + j_;
                }
              }
            }
            y(b,c,i,j) = s;
            max_idx(b,c,i,j) = idx;
          }
        }
      }
    }
  }
#if __NVCC__
  __device__
  void forward_dev(array4<maxB,C,H,W>& x) {
    forward_base(x);
  }
  void forward_gpu(array4<maxB,C,H,W>& x) {
    launch_and_sync((forward_global<<<1,1>>>(dev, x.dev)));
  }
#endif
  void forward_cpu(array4<maxB,C,H,W>& x) {
    forward_base(x);
  }
  array4<maxB,C,H/S,W/S>& forward(array4<maxB,C,H,W>& x) {
    log_start_fun(lgr);
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      forward_cpu(x); break;
#if __NVCC__
    case algo_gpu_base:
      forward_gpu(x); break;
#endif
    default:
      if (opt.gpu_algo) {
#if __NVCC__
        forward_gpu(x);
#else
        err_gpu_algo_no_gpu(opt.algo_s);
#endif
      } else {
        forward_cpu(x);
      }        
    }
    log_end_fun(lgr);
    return y;
  }
  __device__ __host__
  void backward_base(array4<maxB,C,H/S,W/S>& gy) {
    const idx_t B = gy.B;
    gx.set_n_rows(B);
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H/S; i++) {
          for (idx_t j = 0; j < W/S; j++) {
            for (idx_t i_ = S * i; i_ < S * (i + 1); i_++) {
              for (idx_t j_ = S * j; j_ < S * (j + 1); j_++) {
                gx(b,c,i_,j_) = 0;
              }
            }
            idx_t idx = max_idx(b,c,i,j);
            idx_t i_ = idx / W;
            idx_t j_ = idx % W;
            gx(b,c,i_,j_) = gy(b,c,i,j);
          }
        }
      }
    }
  }
#if __NVCC__
  __device__
  void backward_dev(array4<maxB,C,H/S,W/S>& gy) {
    backward_base(gy);
  }
  void backward_gpu(array4<maxB,C,H/S,W/S>& gy) {
    launch_and_sync((backward_global<<<1,1>>>(dev, gy.dev)));
  }
#endif
  void backward_cpu(array4<maxB,C,H/S,W/S>& gy) {
    backward_base(gy);
  }
  array4<maxB,C,H,W>& backward(array4<maxB,C,H/S,W/S>& gy) {
    log_start_fun(lgr);
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      backward_cpu(gy); break;
#if __NVCC__
    case algo_gpu_base:
      backward_gpu(gy); break;
#endif
    default:
      if (opt.gpu_algo) {
#if __NVCC__
        backward_gpu(gy);
#else
        err_gpu_algo_no_gpu(opt.algo_s);
#endif
      } else {
        backward_cpu(gy);
      }        
    }
    log_end_fun(lgr);
    return gx;
  }
  real diff(MaxPooling2D<maxB,C,H,W,S>& b) {
    return y.diff(b.y);
  }
};

template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t S>
  real maxpooling_grad_check_rand(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, idx_t B) {
  /* initialize maxpooling parameters */
  MaxPooling2D<maxB,C,H,W,S> * mp = new MaxPooling2D<maxB,C,H,W,S>();
  mp->init(opt, lgr);
  mp->make_dev();
  mp->to_dev();

  /* make copies */
  MaxPooling2D<maxB,C,H,W,S> * mp_minus = mp->copy();
  MaxPooling2D<maxB,C,H,W,S> * mp_plus  = mp->copy();
  /* make coefficients to make the single loss value */
  array4<maxB,C,H/S,W/S> * alpha = new array4<maxB,C,H/S,W/S>();
  alpha->make_dev(opt.gpu_algo);
  alpha->init_uniform(B, rg, 0.0, 1.0);
  alpha->to_dev();
  /* make input (x) */
  array4<maxB,C,H,W> * x = new array4<maxB,C,H,W>();
  x->make_dev(opt.gpu_algo);
  x->init_uniform(B, rg, 0.0, 1.0);
  x->to_dev();
  /* forward and backward */
  array4<maxB,C,H/S,W/S>& y = mp->forward(*x);
  array4<maxB,C,H,W>& gx = mp->backward(*alpha);
  mp->to_host();

  /* make dx */
  real e = 1.0e-4;
  array4<maxB,C,H,W> * dx = new array4<maxB,C,H,W>();
  dx->init_uniform(B, rg, -e, e);
  /* make x - dx/2 and x + dx/2 */
  array4<maxB,C,H,W> * x_minus = new array4<maxB,C,H,W>(*x);
  x_minus->make_dev(opt.gpu_algo);
  array4<maxB,C,H,W> * x_plus  = new array4<maxB,C,H,W>(*x);
  x_plus->make_dev(opt.gpu_algo);
  x_minus->update(-0.5, *dx);
  x_plus->update( 0.5, *dx);
  x_minus->to_dev();
  x_plus->to_dev();
    
  /* send copies to gpu */
  mp_minus->to_dev();
  mp_plus->to_dev();
  /* make y(x-dx/2), y(x+dx/2) */
  array4<maxB,C,H/S,W/S>& y_minus = mp_minus->forward(*x_minus);
  array4<maxB,C,H/S,W/S>& y_plus  = mp_plus->forward(*x_plus);
  /* get the result back to host */
  y_minus.to_host();
  y_plus.to_host();

  /* get the single loss values */
  real L_minus = alpha->dot(y_minus);
  real L       = alpha->dot(y);
  real L_plus  = alpha->dot(y_plus);
  /* various inner products */
  real gx_gx = gx.dot(gx);                         /* ∂L/∂x・∂L/∂x */
  real dx_dx = dx->dot(*dx);                       /* ∂L/∂x・dx */
  real gx_dx = gx.dot(*dx);                        /* dx・dx */
  real gw_gw = 0;                                  /* ∂L/∂w・∂L/∂w */
  real dw_dw = 0;                                  /* ∂L/∂w・dw */
  real gw_dw = 0;                                  /* dw・dw */

  real rel_e = show_error(gx_gx, dx_dx, gx_dx, gw_gw, dw_dw, gw_dw, L_minus, L, L_plus);
  /* clean up */
  mp->del_dev();
  mp_minus->del_dev();
  mp_plus->del_dev();
  alpha->del_dev();
  x->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();

  delete mp;
  delete mp_minus;
  delete mp_plus;
  delete alpha;
  delete x;
  delete dx;
  delete x_minus;
  delete x_plus;
  return rel_e;
}

int maxpooling_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_sz);
  const idx_t C = 3;
  const idx_t H = 32;
  const idx_t W = 32;
  const idx_t S = 2;
  const int n_checks = opt.iters;
  /* logger */
  logger lgr;
  lgr.start_log(opt);
  /* initialize random number generator */
  rnd_gen_t rg;
  rg.seed(opt.weight_seed);
  /* check errors */
  real max_e = 0.0;
  real sum_e = 0.0;
  for (int iter = 0; iter < n_checks; iter++) {
    printf("==== %d ====\n", iter);
    real e = maxpooling_grad_check_rand<maxB,C,H,W,S>(opt, &lgr, rg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

