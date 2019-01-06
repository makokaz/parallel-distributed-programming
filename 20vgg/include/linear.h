/**
   @file linear.h
 */
#pragma once

#include <math.h>
#include "vgg_util.h"
#include "vgg_arrays.h"

#if __NVCC__
template<idx_t maxB,idx_t IC,idx_t nC>
  struct Linear;

template<idx_t maxB,idx_t IC,idx_t nC>
__global__ void forward_global(Linear<maxB,IC,nC>* dev,
                               array4<maxB,IC,1,1>* x_dev) {
  dev->forward_dev(*x_dev);
}

template<idx_t maxB,idx_t IC,idx_t nC>
  __global__ void backward_global(Linear<maxB,IC,nC>* dev,
                                  array4<maxB,nC,1,1>* gy_dev) {
  dev->backward_dev(*gy_dev);
}

template<idx_t maxB,idx_t IC,idx_t nC>
  __global__ void update_global(Linear<maxB,IC,nC>* dev, real eta) {
  dev->update_dev(eta);
}
#endif

template<idx_t maxB,idx_t IC,idx_t nC>
struct Linear {
#if __NVCC__
  Linear<maxB,IC,nC>* dev;
#endif
  cmdline_opt opt;
  logger * lgr;
  array4<maxB,IC,1,1>* x_ptr;
  array2<IC,nC> w;
  array4<maxB,nC,1,1> y;
  array2<IC,nC> gw;
  array4<maxB,IC,1,1> gx;
  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg) {
    this->opt = opt;
    this->lgr = lgr;
    w.init_normal(IC, rg, 0.0, 1 / sqrt(IC));
  }
  Linear<maxB,IC,nC>* copy() {
    Linear<maxB,IC,nC>* c = new Linear<maxB,IC,nC>(*this);
    c->make_dev();
    return c;
  }
  void set_dev(Linear<maxB,IC,nC>* dev) {
#if __NVCC__
    this->dev = dev;
    w.set_dev(dev ? &dev->w : 0);
    y.set_dev(dev ? &dev->y : 0);
    gw.set_dev(dev ? &dev->gw : 0);
    gx.set_dev(dev ? &dev->gx : 0);
#endif
  }
  void make_dev() {
#if __NVCC__
    if (opt.gpu_algo) {
      dev = (Linear<maxB,IC,nC>*)dev_malloc(sizeof(*this));
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
      Linear<maxB,IC,nC>* dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
  __device__ __host__
  void update_base(real eta) {
    w.update(eta, gw);
  }
#if __NVCC__
  __device__
  void update_dev(real eta) {
    update_base(eta);
  }
  void update_gpu(real eta) {
    launch_and_sync((update_global<<<1,1>>>(dev, eta)));
  }
#endif
  void update_cpu(real eta) {
    update_base(eta);
  }
  void update(real eta) {
    log_start_fun(lgr);
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      update_cpu(eta); break;
#if __NVCC__
    case algo_gpu_base:
      update_gpu(eta); break;
#endif
    default:
      if (opt.gpu_algo) {
#if __NVCC__
        update_gpu(eta);
#else
        err_gpu_algo_no_gpu(opt.algo_s);
#endif
      } else {
        update_cpu(eta);
      }        
    }
    log_end_fun(lgr);
  }
  __device__ __host__
  void forward_base(array4<maxB,IC,1,1>& x) {
    const idx_t B = x.B;
    y.set_n_rows(B);
    x_ptr = &x;
    /* y = x * maxB (x : maxBxIC, w : ICxnC -> y : maxBxnC) */
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < nC; c++) {
        real s = 0.0;
        for (idx_t ic = 0; ic < IC; ic++) {
          s += x(b,ic,0,0) * w(ic,c);
        }
        y(b,c,0,0) = s;
      }
    }
  }
#if __NVCC__
  __device__
  void forward_dev(array4<maxB,IC,1,1>& x) {
    forward_base(x);
  }
  void forward_gpu(array4<maxB,IC,1,1>& x) {
    launch_and_sync((forward_global<<<1,1>>>(dev, x.dev)));
  }
#endif
  void forward_cpu(array4<maxB,IC,1,1>& x) {
    forward_base(x);
  }
  array4<maxB,nC,1,1>& forward(array4<maxB,IC,1,1>& x) {
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
  void backward_base(array4<maxB,nC,1,1>& gy) {
    const idx_t B = gy.B;
    gw.set_n_rows(IC);
    gx.set_n_rows(B);
    array4<maxB,IC,1,1>& x = *x_ptr;
    for (idx_t ic = 0; ic < IC; ic++) {
      for (idx_t c = 0; c < nC; c++) {
        real s = 0.0;
        for (idx_t b = 0; b < B; b++) {
          s += gy(b,c,0,0) * x(b,ic,0,0);
        }
        gw(ic,c) = s;
      }
    }
    for (idx_t b = 0; b < B; b++) {
      for (idx_t ic = 0; ic < IC; ic++) {
        real s = 0.0;
        for (idx_t c = 0; c < nC; c++) {
          s += gy(b,c,0,0) * w(ic,c);
        }
        gx(b,ic,0,0) = s;
      }
    }
  }
#if __NVCC__
  __device__
  void backward_dev(array4<maxB,nC,1,1>& gy) {
    backward_base(gy);
  }
  void backward_gpu(array4<maxB,nC,1,1>& gy) {
    launch_and_sync((backward_global<<<1,1>>>(dev, gy.dev)));
  }
#endif
  void backward_cpu(array4<maxB,nC,1,1>& gy) {
    backward_base(gy);
  }
  array4<maxB,IC,1,1>& backward(array4<maxB,nC,1,1>& gy) {
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
  real diff(Linear<maxB,IC,nC>& b) {
    return y.diff(b.y);
  }
  void rand_grad(rnd_gen_t& rg, real p, real q) {
    gw.init_uniform(IC, rg, p, q);
  }
  void set_grad(Linear<maxB,IC,nC>& o) {
    gw = o.gw;
  }
  real gw_dot_gw(Linear<maxB,IC,nC>& o) {
    return gw.dot(o.gw);
  }
};

template<idx_t maxB,idx_t IC,idx_t nC>
  real linear_grad_check_rand(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, idx_t B) {
  /* initialize linear parameters */
  Linear<maxB,IC,nC> * linear = new Linear<maxB,IC,nC>();
  linear->init(opt, lgr, rg);
  linear->make_dev();
  linear->to_dev();
  /* make w - dw/2 and w + dw/2 */
  Linear<maxB,IC,nC> * linear_minus = linear->copy();
  Linear<maxB,IC,nC> * linear_plus  = linear->copy();
  /* make coefficients to make the single loss value */
  array4<maxB,nC,1,1> * alpha = new array4<maxB,nC,1,1>();
  alpha->make_dev(opt.gpu_algo);
  alpha->init_uniform(B, rg, 0.0, 1.0);
  alpha->to_dev();
  /* make input (x) */
  array4<maxB,IC,1,1> * x = new array4<maxB,IC,1,1>();
  x->make_dev(opt.gpu_algo);
  x->init_uniform(B, rg, 0.0, 1.0);
  x->to_dev();
  /* forward and backward */
  array4<maxB,nC,1,1>& y = linear->forward(*x);
  array4<maxB,IC,1,1>& gx = linear->backward(*alpha);
  /* ensure the gradient is back to host */
  linear->to_host();

  /* make dx */
  real e = 1.0e-4;
  array4<maxB,IC,1,1> * dx = new array4<maxB,IC,1,1>();
  dx->init_uniform(B, rg, -e, e);
  /* make x - dx/2 and x + dx/2 */
  array4<maxB,IC,1,1> * x_minus = new array4<maxB,IC,1,1>(*x);
  x_minus->make_dev(opt.gpu_algo);
  array4<maxB,IC,1,1> * x_plus  = new array4<maxB,IC,1,1>(*x);
  x_plus->make_dev(opt.gpu_algo);
  /* update on the host and send the to gpu */
  x_minus->update(-0.5, *dx);
  x_plus->update( 0.5, *dx);
  x_minus->to_dev();
  x_plus->to_dev();
    
  /* set gw to a random vector */
  linear_minus->rand_grad(rg, -e, e);
  linear_plus->set_grad(*linear_minus);
  /* send them to gpu */
  linear_minus->to_dev();
  linear_plus->to_dev();
  /* update weights using gw (update runs on gpu) */
  linear_minus->update(-0.5);      /* w -= dw/2 */
  linear_plus->update(0.5);        /* w += dw/2 */
  /* make y(w-dw/2,x-dx/2), y(w+dw/2,x+dx/2) */
  array4<maxB,nC,1,1>& y_minus = linear_minus->forward(*x_minus);
  array4<maxB,nC,1,1>& y_plus  = linear_plus->forward(*x_plus);
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
  real gw_gw = linear->gw_dot_gw(*linear);             /* ∂L/∂w・∂L/∂w */
  real dw_dw = linear_minus->gw_dot_gw(*linear_minus); /* ∂L/∂w・dw */
  real gw_dw = linear->gw_dot_gw(*linear_minus);       /* dw・dw */

  real rel_e = show_error(gx_gx, dx_dx, gx_dx, gw_gw, dw_dw, gw_dw, L_minus, L, L_plus);
  /* clean up */
  linear->del_dev();
  linear_minus->del_dev();
  linear_plus->del_dev();
  alpha->del_dev();
  x->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();

  delete linear;
  delete linear_minus;
  delete linear_plus;
  delete alpha;
  delete x;
  delete dx;
  delete x_minus;
  delete x_plus;
  return rel_e;
}

int linear_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_sz);
  const idx_t IC = 512;
  const idx_t nC = 10;
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
    real e = linear_grad_check_rand<maxB,IC,nC>(opt, &lgr, rg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}
