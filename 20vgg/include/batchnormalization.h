/**
   @file batchnormalization.h
 */
#pragma once

#include <math.h>
#include "vgg_util.h"
#include "vgg_arrays.h"

#if __NVCC__
template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
  struct BatchNormalization;

template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
__global__ void forward_global(BatchNormalization<maxB,IC,H,W>* dev,
                               array4<maxB,IC,H,W>* x_dev) {
  dev->forward_dev(*x_dev);
}

template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
__global__ void backward_global(BatchNormalization<maxB,IC,H,W>* dev,
                                array4<maxB,IC,H,W>* gy_dev) {
  dev->backward_dev(*gy_dev);
}

template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
  __global__ void update_global(BatchNormalization<maxB,IC,H,W>* dev, real eta) {
  dev->update_dev(eta);
}
#endif

template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
struct BatchNormalization {
#if __NVCC__
  BatchNormalization<maxB,IC,H,W> * dev;
#endif
  cmdline_opt opt;
  logger * lgr;
  vec<IC> gamma;
  vec<IC> beta;
  array4<maxB,IC,H,W> x_hat;
  vec<IC> inv_std;
  array4<maxB,IC,H,W> y;
  vec<IC> ggamma;
  vec<IC> gbeta;
  array4<maxB,IC,H,W> gx;

  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg) {
    this->opt = opt;
    this->lgr = lgr;
    gamma.init_uniform(IC, rg, 0.0, 1.0);
    beta.init_uniform(IC, rg, 0.0, 1.0);
  }
  BatchNormalization<maxB,IC,H,W>* copy() {
    BatchNormalization<maxB,IC,H,W>* c = new BatchNormalization<maxB,IC,H,W>(*this);
    c->make_dev();
    return c;
  }
  void set_dev(BatchNormalization<maxB,IC,H,W>* dev) {
#if __NVCC__
    this->dev = dev;
    gamma.set_dev(dev ? &dev->gamma : 0);
    beta.set_dev(dev ? &dev->beta : 0);
    x_hat.set_dev(dev ? &dev->x_hat : 0);
    inv_std.set_dev(dev ? &dev->inv_std : 0);
    y.set_dev(dev ? &dev->y : 0);
    ggamma.set_dev(dev ? &dev->ggamma : 0);
    gbeta.set_dev(dev ? &dev->gbeta : 0);
    gx.set_dev(dev ? &dev->gx : 0);
#endif
  }
  void make_dev() {
#if __NVCC__
    if (opt.gpu_algo) {
      dev = (BatchNormalization<maxB,IC,H,W>*)dev_malloc(sizeof(*this));
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
      BatchNormalization<maxB,IC,H,W>* dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
  __device__ __host__
  void update_base(real eta) {
    gamma.update(eta, ggamma);
    beta.update(eta, gbeta);
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
  vec<IC> mean_bij(array4<maxB,IC,H,W>& x) {
    const idx_t B = x.B;
    vec<IC> mean;
    mean.init_const(IC, 0);
    for (idx_t ic = 0; ic < IC; ic++) {
      real s = 0.0;
      for (idx_t b = 0; b < B; b++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            s += x(b,ic,i,j);
          }
        }
      }
      mean(ic) = s / (B * H * W);
    }
    return mean;
  }

  __device__ __host__
  vec<IC>& inv_std_bij(array4<maxB,IC,H,W>& x, vec<IC>& mu) {
    const idx_t B = x.B;
    const real epsilon = 2.0e-5;
    const real l_BHW = 1 / (real)(B * H * W);
    inv_std.set_n(IC);
    for (idx_t ic = 0; ic < IC; ic++) {
      real s = 0.0;
      for (idx_t b = 0; b < B; b++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            real ds = x(b,ic,i,j) - mu(ic);
            s += ds * ds;
          }
        }
      }
      inv_std(ic) = 1.0 / sqrt(s * l_BHW + epsilon);
    }
    return inv_std;
  }
  __device__ __host__ 
  void forward_base(array4<maxB,IC,H,W>& x) {
    const idx_t B = x.B;
    x_hat.set_n_rows(B);
    y.set_n_rows(B);
    if (B * H * W > 1) {
      vec<IC> mu = mean_bij(x);
      inv_std = inv_std_bij(x, mu);
      for (idx_t b = 0; b < B; b++) {
        for (idx_t ic = 0; ic < IC; ic++) {
          for (idx_t i = 0; i < H; i++) {
            for (idx_t j = 0; j < W; j++) {
              x_hat(b,ic,i,j) = (x(b,ic,i,j) - mu(ic)) * inv_std(ic);
              y(b,ic,i,j) = gamma(ic) * x_hat(b,ic,i,j) + beta(ic);
            }
          }
        }
      }
    } else {
      for (idx_t b = 0; b < B; b++) {
        for (idx_t ic = 0; ic < IC; ic++) {
          for (idx_t i = 0; i < H; i++) {
            for (idx_t j = 0; j < W; j++) {
              y(b,ic,i,j) = x(b,ic,i,j);
            }
          }
        }
      }
    }
  }
#if __NVCC__
  __device__
  void forward_dev(array4<maxB,IC,H,W>& x) {
    forward_base(x);
  }
  void forward_gpu(array4<maxB,IC,H,W>& x) {
    launch_and_sync((forward_global<<<1,1>>>(dev, x.dev)));
  }
#endif
  void forward_cpu(array4<maxB,IC,H,W>& x) {
    forward_base(x);
  }
  array4<maxB,IC,H,W>& forward(array4<maxB,IC,H,W>& x) {
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
  void backward_base(array4<maxB,IC,H,W>& gy) {
    const idx_t B = gy.B;
    gx.set_n_rows(B);
    gbeta.set_n(IC);
    ggamma.set_n(IC);
    if (B * H * W > 1) {
      for (idx_t ic = 0; ic < IC; ic++) {
        real s = 0.0, t = 0.0;
        for (idx_t b = 0; b < B; b++) {
          for (idx_t i = 0; i < H; i++) {
            for (idx_t j = 0; j < W; j++) {
              s += gy(b,ic,i,j);
              t += gy(b,ic,i,j) * x_hat(b,ic,i,j);
            }
          }
        }
        gbeta(ic) = s;
        ggamma(ic) = t;
      }
      real l_BHW = 1 / (real)(B * H * W);
      for (idx_t ic = 0; ic < IC; ic++) {
        real a = gamma(ic) * inv_std(ic);
        real gg = ggamma(ic);
        real gb = gbeta(ic);
        for (idx_t b = 0; b < B; b++) {
          for (idx_t i = 0; i < H; i++) {
            for (idx_t j = 0; j < W; j++) {
              gx(b,ic,i,j) = a * (gy(b,ic,i,j) - l_BHW * (gg * x_hat(b,ic,i,j) + gb));
            }
          }
        }
      }
    } else {
      for (idx_t b = 0; b < B; b++) {
        for (idx_t ic = 0; ic < IC; ic++) {
          for (idx_t i = 0; i < H; i++) {
            for (idx_t j = 0; j < W; j++) {
              gx(b,ic,i,j) = gy(b,ic,i,j);
            }
          }
        }
      }
    }
  }
#if __NVCC__
  __device__
  void backward_dev(array4<maxB,IC,H,W>& gy) {
    backward_base(gy);
  }
  void backward_gpu(array4<maxB,IC,H,W>& gy) {
    launch_and_sync((backward_global<<<1,1>>>(dev, gy.dev)));
  }
#endif
  void backward_cpu(array4<maxB,IC,H,W>& gy) {
    backward_base(gy);
  }
  array4<maxB,IC,H,W>& backward(array4<maxB,IC,H,W>& gy) {
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
  real diff(BatchNormalization<maxB,IC,H,W>& b) {
    return y.diff(b.y);
  }
  void rand_grad(rnd_gen_t& rg, real p, real q) {
    ggamma.init_uniform(IC, rg, p, q);
    gbeta.init_uniform(IC, rg, p, q);
  }
  void set_grad(BatchNormalization<maxB,IC,H,W>& o) {
    ggamma = o.ggamma;
    gbeta = o.gbeta;
  }
  real gw_dot_gw(BatchNormalization<maxB,IC,H,W>& b) {
    BatchNormalization<maxB,IC,H,W>& a = *this;
    real s = 0.0;
    s += a.ggamma.dot(b.ggamma);
    s += a.gbeta.dot(b.gbeta);
    return s;
  }
};

template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
  static real batchnormalization_grad_check_rand(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, idx_t B) {
  /* initialize batch normalization parameters */
  BatchNormalization<maxB,IC,H,W> * bn = new BatchNormalization<maxB,IC,H,W>();
  bn->init(opt, lgr, rg);
  bn->make_dev();
  bn->to_dev();
  /* make w - dw/2 and w + dw/2 */
  BatchNormalization<maxB,IC,H,W> * bn_minus = bn->copy();
  BatchNormalization<maxB,IC,H,W> * bn_plus = bn->copy();
  /* make coefficients to make the single loss value */
  array4<maxB,IC,H,W> * alpha = new array4<maxB,IC,H,W>();
  alpha->make_dev(opt.gpu_algo);
  alpha->init_uniform(B, rg, 0.0, 1.0);
  alpha->to_dev();
  /* make input (x) */
  array4<maxB,IC,H,W> * x = new array4<maxB,IC,H,W>();
  x->make_dev(opt.gpu_algo);
  x->init_uniform(B, rg, 0.0, 1.0);
  x->to_dev();
  /* forward and backward */
  array4<maxB,IC,H,W>& y = bn->forward(*x);
  array4<maxB,IC,H,W>& gx = bn->backward(*alpha);
  /* ensure the gradient is back to host */
  bn->to_host();

  /* make dx */
  real e = 1.0e-4;
  array4<maxB,IC,H,W> * dx = new array4<maxB,IC,H,W>();
  dx->init_uniform(B, rg, -e, e);
  /* make x - dx/2 and x + dx/2 */
  array4<maxB,IC,H,W> * x_minus = new array4<maxB,IC,H,W>(*x);
  x_minus->make_dev(opt.gpu_algo);
  array4<maxB,IC,H,W> * x_plus  = new array4<maxB,IC,H,W>(*x);
  x_plus->make_dev(opt.gpu_algo);
  /* update on the host and send the to gpu */
  x_minus->update(-0.5, *dx);
  x_plus->update( 0.5, *dx);
  x_minus->to_dev();
  x_plus->to_dev();
    
  /* set gw to a random vector */
  bn_minus->rand_grad(rg, -e, e);
  bn_plus->set_grad(*bn_minus);
  /* send them to gpu */
  bn_minus->to_dev();
  bn_plus->to_dev();
  /* update weights using gw (update runs on gpu) */
  bn_minus->update(-0.5);      /* w -= dw/2 */
  bn_plus->update(0.5);        /* w += dw/2 */
  /* make y(w-dw,x-dx), y(w+dw,x+dx) */
  array4<maxB,IC,H,W>& y_minus = bn_minus->forward(*x_minus);
  array4<maxB,IC,H,W>& y_plus  = bn_plus->forward(*x_plus);
  /* get the result back to host */
  y_minus.to_host();
  y_plus.to_host();

  /* get the single loss values */
  real L_minus = alpha->dot(y_minus);
  real L       = alpha->dot(y);
  real L_plus  = alpha->dot(y_plus);
  /* various inner products */
  real gx_gx = gx.dot(gx);                       /* ∂L/∂x・∂L/∂x */
  real dx_dx = dx->dot(*dx);                     /* ∂L/∂x・dx */
  real gx_dx = gx.dot(*dx);                      /* dx・dx */
  real gw_gw = bn->gw_dot_gw(*bn);               /* ∂L/∂w・∂L/∂w */
  real dw_dw = bn_minus->gw_dot_gw(*bn_minus);   /* ∂L/∂w・dw */
  real gw_dw = bn->gw_dot_gw(*bn_minus);         /* dw・dw */

  real rel_e = show_error(gx_gx, dx_dx, gx_dx, gw_gw, dw_dw, gw_dw, L_minus, L, L_plus);
  /* clean up */
  bn_minus->del_dev();
  bn_plus->del_dev();
  alpha->del_dev();
  x->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();
  
  delete bn_minus;
  delete bn_plus;
  delete alpha;
  delete x;
  delete dx;
  delete x_minus;
  delete x_plus;
  return rel_e;
}

int batchnormalization_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_sz);
  const idx_t IC = 64;
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
    real e = batchnormalization_grad_check_rand<maxB,IC,H,W>(opt, &lgr, rg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

