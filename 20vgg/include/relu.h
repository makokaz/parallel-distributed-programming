/**
   @file relu.h
 */
#pragma once

#include "vgg_util.h"
#include "vgg_arrays.h"

#if __NVCC__
template<idx_t maxB,idx_t C,idx_t H,idx_t W>
  struct Relu;

template<idx_t maxB,idx_t C,idx_t H,idx_t W>
  __global__ void forward_global(Relu<maxB,C,H,W>* dev,
                                 array4<maxB,C,H,W>* x_dev) {
  dev->forward_dev(*x_dev);
}

template<idx_t maxB,idx_t C,idx_t H,idx_t W>
  __global__ void backward_global(Relu<maxB,C,H,W>* dev,
                                  array4<maxB,C,H,W>* gy_dev) {
  dev->backward_dev(*gy_dev);
}
#endif

template<idx_t maxB,idx_t C,idx_t H,idx_t W>
struct Relu {
#if __NVCC__
  Relu<maxB,C,H,W>* dev;
#endif
  cmdline_opt opt;
  logger * lgr;
  array4<maxB,C,H,W>* x_ptr;
  array4<maxB,C,H,W> y;
  array4<maxB,C,H,W> gx;
  void init(cmdline_opt opt, logger * lgr) {
    this->opt = opt;
    this->lgr = lgr;
  }
  Relu<maxB,C,H,W>* copy() {
    Relu<maxB,C,H,W>* c = new Relu<maxB,C,H,W>(*this);
    c->make_dev();
    return c;
  }
  void set_dev(Relu<maxB,C,H,W>* dev_) {
#if __NVCC__
    dev = dev_;
    y.set_dev(dev ? &dev->y : 0);
    gx.set_dev(dev ? &dev->gx : 0);
#endif
  }
  void make_dev() {
#if __NVCC__
    if (opt.gpu_algo) {
      dev = (Relu<maxB,C,H,W>*)dev_malloc(sizeof(*this));
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
      Relu<maxB,C,H,W>* dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
  __device__ __host__
  void forward_base(array4<maxB,C,H,W>& x) {
    const idx_t B = x.B;
    y.set_n_rows(B);
    x_ptr = &x;
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            y(b,c,i,j) = max_r(0, x(b,c,i,j));
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
  array4<maxB,C,H,W>& forward(array4<maxB,C,H,W>& x) {
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
  void backward_base(array4<maxB,C,H,W>& gy) {
    const idx_t B = gy.B;
    gx.set_n_rows(B);
    array4<maxB,C,H,W>& x = *x_ptr;
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            gx(b,c,i,j) = (x(b,c,i,j) >= 0 ? gy(b,c,i,j) : 0);
          }
        }
      }
    }
  }
#if __NVCC__
  __device__
  void backward_dev(array4<maxB,C,H,W>& gy) {
    backward_base(gy);
  }
  void backward_gpu(array4<maxB,C,H,W>& gy) {
    launch_and_sync((backward_global<<<1,1>>>(dev, gy.dev)));
  }
#endif
  void backward_cpu(array4<maxB,C,H,W>& gy) {
    backward_base(gy);
  }
  array4<maxB,C,H,W>& backward(array4<maxB,C,H,W>& gy) {
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
  real diff(Relu<maxB,C,H,W>& b) {
    return y.diff(b.y);
  }
};

template<idx_t maxB,idx_t C,idx_t H,idx_t W>
  static real relu_grad_check_rand(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, idx_t B) {
  /* initialize relu parameters */
  Relu<maxB,C,H,W> * relu = new Relu<maxB,C,H,W>();
  relu->init(opt, lgr);
  relu->make_dev();
  relu->to_dev();
  /* make copies */
  Relu<maxB,C,H,W> * relu_minus = relu->copy();
  Relu<maxB,C,H,W> * relu_plus  = relu->copy();
  /* make coefficients to make the single loss value */
  array4<maxB,C,H,W> * alpha = new array4<maxB,C,H,W>();
  alpha->make_dev(opt.gpu_algo);
  alpha->init_uniform(B, rg, 0.0, 1.0);
  alpha->to_dev();
  /* make input (x) */
  array4<maxB,C,H,W> * x = new array4<maxB,C,H,W>();
  x->make_dev(opt.gpu_algo);
  x->init_uniform(B, rg, 0.0, 1.0);
  x->to_dev();
  /* forward and backward */
  array4<maxB,C,H,W>& y = relu->forward(*x);
  array4<maxB,C,H,W>& gx = relu->backward(*alpha);
  relu->to_host();
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
  x_plus->update(0.5, *dx);
  x_minus->to_dev();
  x_plus->to_dev();
    
  /* send copies to gpu */
  relu_minus->to_dev();
  relu_plus->to_dev();
  /* make y(x-dx/2), y(x+dx/2) */
  array4<maxB,C,H,W>& y_minus = relu_minus->forward(*x_minus);
  array4<maxB,C,H,W>& y_plus  = relu_plus->forward(*x_plus);
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
  relu->del_dev();
  relu_minus->del_dev();
  relu_plus->del_dev();
  alpha->del_dev();
  x->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();

  delete relu;
  delete relu_minus;
  delete relu_plus;
  delete alpha;
  delete x;
  delete dx;
  delete x_minus;
  delete x_plus;
  return rel_e;
}

int relu_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_sz);
  const idx_t C = 3;
  const idx_t H = 32;
  const idx_t W = 32;
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
    real e = relu_grad_check_rand<maxB,C,H,W>(opt, &lgr, rg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

