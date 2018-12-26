/**
   @file batchnormalization.h
 */
#pragma once

#include <math.h>
#include "vgg_util.h"
#include "vgg_arrays.h"

template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
struct BatchNormalization {
  cmdline_opt opt;
  idx_t B;
  vec<IC> gamma;
  vec<IC> beta;
  array4<maxB,IC,H,W> x_hat;
  vec<IC> inv_std;
  array4<maxB,IC,H,W> y;
  vec<IC> ggamma;
  vec<IC> gbeta;
  array4<maxB,IC,H,W> gx;

  void init(cmdline_opt opt_, idx_t B_, rnd_gen_t& rg) {
    opt = opt_;
    B = B_;
    assert(B <= maxB);
    gamma.init_uniform(IC, rg, 0.0, 1.0);
    beta.init_uniform(IC, rg, 0.0, 1.0);
    x_hat.init(B);
    inv_std.init(IC);
    y.init(B);
    ggamma.init(IC);
    gbeta.init(IC);
    gx.init(B);
  }

  void update(real eta) {
    gamma.update(eta, ggamma);
    beta.update(eta, gbeta);
  }

  vec<IC> mean_bij(array4<maxB,IC,H,W>& x) {
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

  vec<IC>& inv_std_bij(array4<maxB,IC,H,W>& x, vec<IC>& mu) {
    const real epsilon = 2.0e-5;
    const real l_BHW = 1 / (real)(B * H * W);
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

  array4<maxB,IC,H,W>& forward(array4<maxB,IC,H,W>& x) {
    if (opt.verbose>=2) {
      printf("BatchNormalization<maxB=%ld,IC=%ld,H=%ld,W=%ld>.forward(B=%ld) starts\n",
             (long)maxB, (long)IC, (long)H, (long)W, (long)B);
    }
    tsc_t t0 = get_tsc();
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
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("BatchNormalization<maxB=%ld,IC=%ld,H=%ld,W=%ld>.forward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)IC, (long)H, (long)W, (long)B, t1.ns - t0.ns);
    }
    return y;
  }
  array4<maxB,IC,H,W>& backward(array4<maxB,IC,H,W>& gy) {
    if (opt.verbose>=2) {
      printf("BatchNormalization<maxB=%ld,IC=%ld,H=%ld,W=%ld>.backward(B=%ld) starts\n",
             (long)maxB, (long)IC, (long)H, (long)W, (long)B);
    }
    tsc_t t0 = get_tsc();
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
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("BatchNormalization<maxB=%ld,IC=%ld,H=%ld,W=%ld>.backward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)IC, (long)H, (long)W, (long)B, t1.ns - t0.ns);
    }
    return gx;
  }
};

int batchnormalization_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  rnd_gen_t rg;
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = opt.batch_sz;
  const idx_t IC = 64;
  const idx_t H = 32;
  const idx_t W = 32;
  rg.seed(opt.weight_seed);
  BatchNormalization<maxB,IC,H,W> bn;
  array4<maxB,IC,H,W> x;
  bn.init(opt, B, rg);
  x.init_uniform(B, rg, 0.0, 1.0);
  bn.forward(x);
  return 0;
}

