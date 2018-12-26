/**
   @file linear.h
 */
#pragma once

#include <math.h>
#include "vgg_util.h"
#include "vgg_arrays.h"

template<idx_t maxB,idx_t IC,idx_t nC>
struct Linear {
  cmdline_opt opt;
  idx_t B;
  array4<maxB,IC,1,1>* x_ptr;
  array2<IC,nC> w;
  array4<maxB,nC,1,1> y;
  array2<IC,nC> gw;
  array4<maxB,IC,1,1> gx;
  void init(cmdline_opt opt_, idx_t B_, rnd_gen_t& rg) {
    opt = opt_;
    B = B_;
    assert(B <= maxB);
    w.init_normal(IC, rg, 0.0, 1 / sqrt(IC));
    y.init(B);
    gw.init(IC);
    gx.init(B);
  }
  void update(real eta) {
    w.update(eta, gw);
  }
  array4<maxB,nC,1,1>& forward(array4<maxB,IC,1,1>& x) {
    if (opt.verbose>=2) {
      printf("Linear<maxB=%ld,IC=%ld,nC=%ld>.forward(B=%ld)\n",
             (long)maxB, (long)IC, (long)nC, (long)B);
    }
    tsc_t t0 = get_tsc();
    /* y = x * maxB (x : maxBxIC, w : ICxnC -> y : maxBxnC) */
    x_ptr = &x;
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < nC; c++) {
        real s = 0.0;
        for (idx_t ic = 0; ic < IC; ic++) {
          s += x(b,ic,0,0) * w(ic,c);
        }
        y(b,c,0,0) = s;
      }
    }
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("Linear<maxB=%ld,IC=%ld,nC=%ld>.forward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)IC, (long)nC, (long)B, t1.ns - t0.ns);
    }
    return y;
  }
  array4<maxB,IC,1,1>& backward(array4<maxB,nC,1,1>& gy) {
    if (opt.verbose>=2) {
      printf("Linear<maxB=%ld,IC=%ld,nC=%ld>.backward\n",
             (long)maxB, (long)IC, (long)nC);
    }
    tsc_t t0 = get_tsc();
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
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("Linear<maxB=%ld,IC=%ld,nC=%ld>.backward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)IC, (long)nC, (long)B, t1.ns - t0.ns);
    }
    return gx;
  }
};

int linear_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = opt.batch_sz;
  const idx_t IC = 3;
  const idx_t nC = 10;
  rnd_gen_t rg;
  rg.seed(opt.weight_seed);

  Linear<maxB,IC,nC> linear;
  array4<maxB,IC,1,1> x;
  linear.init(opt, B, rg);
  x.init_uniform(B, rg, 0.0, 1.0);
  linear.forward(x);
  return 0;
}
