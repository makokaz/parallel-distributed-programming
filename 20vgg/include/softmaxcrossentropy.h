/**
   @file softmaxcrossentropy.h
 */
#pragma once

#include <math.h>
#include "vgg_util.h"
#include "vgg_arrays.h"

#if __NVCC__
template<idx_t maxB,idx_t nC>
  struct SoftmaxCrossEntropy;

template<idx_t maxB,idx_t nC>
  __global__ void forward_global(SoftmaxCrossEntropy<maxB,nC>* dev,
                                 array4<maxB,nC,1,1>* x_dev, ivec<maxB>* t_dev) {
  dev->forward_dev(*x_dev, *t_dev);
}

template<idx_t maxB,idx_t nC>
  __global__ void backward_global(SoftmaxCrossEntropy<maxB,nC>* dev,
                                  vec<maxB>* gy_dev) {
  dev->backward_dev(*gy_dev);
}
#endif

template<idx_t maxB,idx_t nC>
struct SoftmaxCrossEntropy {
#if __NVCC__
  SoftmaxCrossEntropy<maxB,nC>* dev;
#endif
  logger * lgr;
  cmdline_opt opt;
  ivec<maxB>* t_ptr;
  array2<maxB,nC> lsm;
  vec<maxB> y;
  array4<maxB,nC,1,1> gx;

  void init(cmdline_opt opt, logger * lgr) {
    this->opt = opt;
    this->lgr = lgr;
  }
  SoftmaxCrossEntropy<maxB,nC>* copy() {
    SoftmaxCrossEntropy<maxB,nC>* c = new SoftmaxCrossEntropy<maxB,nC>(*this);
    c->make_dev();
    return c;
  }
  void set_dev(SoftmaxCrossEntropy<maxB,nC>* dev) {
#if __NVCC__
    this->dev = dev;
    y.set_dev(dev ? &dev->y : 0);
    gx.set_dev(dev ? &dev->gx : 0);
#endif
  }
  void make_dev() {
#if __NVCC__
    if (opt.gpu_algo) {
      dev = (SoftmaxCrossEntropy<maxB,nC>*)dev_malloc(sizeof(*this));
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
      SoftmaxCrossEntropy<maxB,nC>* dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }

  /* for a column vector x = (x_0, ..., x_{n-1}), 

                           (exp(x_0)     / Σ_j exp(x_j))
     logsoftmax(x)_i = log (exp(x_1)     / Σ_j exp(x_j))
                           (   ...       / Σ_j exp(x_j))
                           (exp(x_{n-1}) / Σ_j exp(x_j))

 */
  __device__ __host__
  array2<maxB,nC>& logsoftmax(array4<maxB,nC,1,1>& x) {
    const idx_t B = x.B;
    for (long b = 0; b < B; b++) {
      long m = 0;
      for (long c = 0; c < nC; c++) {
        m = (x(b,m,0,0) < x(b,c,0,0) ? c : m);
      }
      real s = 0.0;
      for (long c = 0; c < nC; c++) {
        lsm(b,c) = x(b,c,0,0) - x(b,m,0,0);
        s += exp(lsm(b,c));
      }
      for (long c = 0; c < nC; c++) {
        lsm(b,c) -= log(s);
      }
    }
    return lsm;
  }
  __device__ __host__
  void forward_base(array4<maxB,nC,1,1>& x, ivec<maxB>& t) {
    const idx_t B = x.B;
    lsm.set_n_rows(B);
    y.set_n(B);
    t_ptr = &t;

    logsoftmax(x);
    for (idx_t b = 0; b < B; b++) {
      y(b) = -lsm(b,t(b));
    }
  }
#if __NVCC__
  __device__
  void forward_dev(array4<maxB,nC,1,1>& x, ivec<maxB>& t) {
    forward_base(x, t);
  }
  void forward_gpu(array4<maxB,nC,1,1>& x, ivec<maxB>& t) {
    launch_and_sync((forward_global<<<1,1>>>(dev, x.dev, t.dev)));
  }
#endif
  void forward_cpu(array4<maxB,nC,1,1>& x, ivec<maxB>& t) {
    forward_base(x, t);
  }
  vec<maxB>& forward(array4<maxB,nC,1,1>& x, ivec<maxB>& t) {
    log_start_fun(lgr);
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      forward_cpu(x, t); break;
#if __NVCC__
    case algo_gpu_base:
      forward_gpu(x, t); break;
#endif
    default:
      if (opt.gpu_algo) {
#if __NVCC__
        forward_gpu(x, t);
#else
        err_gpu_algo_no_gpu(opt.algo_s);
#endif
      } else {
        forward_cpu(x, t);
      }        
    }
    log_end_fun(lgr);
    return y;
  }
  __device__ __host__
  void backward_base(vec<maxB>& gy) {
    const idx_t B = gy.n;
    gx.set_n_rows(B);
    ivec<maxB>& t = *t_ptr;
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < nC; c++) {
        if (c == t(b)) {
          gx(b,c,0,0) = gy(b) * (-1 + exp(lsm(b,c)));
        } else {
          gx(b,c,0,0) = gy(b) * exp(lsm(b,c));
        }
      }
    }
  }
#if __NVCC__
  __device__
  void backward_dev(vec<maxB>& gy) {
    backward_base(gy);
  }
  void backward_gpu(vec<maxB>& gy) {
    launch_and_sync((backward_global<<<1,1>>>(dev, gy.dev)));
  }
#endif
  void backward_cpu(vec<maxB>& gy) {
    backward_base(gy);
  }
  array4<maxB,nC,1,1>& backward(vec<maxB>& gy) {
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
  real diff(SoftmaxCrossEntropy<maxB,nC>& b) {
    return y.diff(b.y);
  }
};

template<idx_t maxB,idx_t nC>
  static real softmaxcrossentropy_grad_check_rand(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, idx_t B) {
  /* initialize softmax */
  SoftmaxCrossEntropy<maxB,nC> * smxe = new SoftmaxCrossEntropy<maxB,nC>();
  smxe->init(opt, lgr);
  smxe->make_dev();
  smxe->to_dev();
  /* make copies */
  SoftmaxCrossEntropy<maxB,nC>* smxe_minus = smxe->copy();
  SoftmaxCrossEntropy<maxB,nC>* smxe_plus  = smxe->copy();
  /* make coefficients to make the single loss value */
  vec<maxB> * alpha = new vec<maxB>();
  alpha->make_dev(opt.gpu_algo);
  alpha->init_uniform(B, rg, 0.0, 1.0);
  alpha->to_dev();
  /* make input (x) */
  array4<maxB,nC,1,1> * x = new array4<maxB,nC,1,1>();
  x->make_dev(opt.gpu_algo);
  x->init_uniform(B, rg, 0.0, 1.0);
  x->to_dev();
  /* make input (t) */
  ivec<maxB> * t = new ivec<maxB>();
  t->make_dev(opt.gpu_algo);
  t->init_uniform(B, rg, 0, nC);
  t->to_dev();
  /* forward and backward */
  vec<maxB>& y = smxe->forward(*x, *t);
  array4<maxB,nC,1,1>& gx = smxe->backward(*alpha);
  smxe->to_host();

  /* make dx */
  real e = 1.0e-4;
  array4<maxB,nC,1,1> * dx = new array4<maxB,nC,1,1>();
  dx->init_uniform(B, rg, -e, e);
  /* make x - dx/2 and x + dx/2 */
  array4<maxB,nC,1,1> * x_minus = new array4<maxB,nC,1,1>(*x);
  x_minus->make_dev(opt.gpu_algo);
  array4<maxB,nC,1,1> * x_plus  = new array4<maxB,nC,1,1>(*x);
  x_plus->make_dev(opt.gpu_algo);
  x_minus->update(-0.5, *dx);
  x_plus->update( 0.5, *dx);
  x_minus->to_dev();
  x_plus->to_dev();
    
  /* send copies to gpu */
  smxe_minus->to_dev();
  smxe_plus->to_dev();
  /* make y(x-dx/2), y(x+dx/2) */
  vec<maxB>& y_minus = smxe_minus->forward(*x_minus, *t);
  vec<maxB>& y_plus  = smxe_plus->forward(*x_plus, *t);
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
  smxe->del_dev();
  smxe_minus->del_dev();
  smxe_plus->del_dev();
  alpha->del_dev();
  x->del_dev();
  t->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();

  delete smxe;
  delete smxe_minus;
  delete smxe_plus;
  delete alpha;
  delete x;
  delete t;
  delete dx;
  delete x_minus;
  delete x_plus;
  return rel_e;
}

int softmaxcrossentropy_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_sz);
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
    real e = softmaxcrossentropy_grad_check_rand<maxB,nC>(opt, &lgr, rg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

