/**
   @file softmaxcrossentropy.h
 */
#pragma once

#include <math.h>
#include "vgg_util.h"
#include "vgg_arrays.h"

template<idx_t maxB,idx_t nC>
struct SoftmaxCrossEntropy {
  cmdline_opt opt;
  idx_t B;
  ivec<maxB>* t_ptr;
  array2<maxB,nC> lsm;
  vec<maxB> y;
  array4<maxB,nC,1,1> gx;

  void init(cmdline_opt opt_, idx_t B_) {
    opt = opt_;
    B = B_;
    assert(B <= maxB);
    lsm.init(B);
    y.init(B);
    gx.init(B);
  }
  
  array2<maxB,nC>& logsoftmax(array4<maxB,nC,1,1>& x) {
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

  vec<maxB>& forward(array4<maxB,nC,1,1>& x, ivec<maxB>& t) {
    if (opt.verbose>=2) {
      printf("SoftmaxCrossEntropy<maxB=%ld,nC=%ld>.forward(B=%ld) starts\n",
             (long)maxB, (long)nC, (long)B);
    }
    tsc_t t0 = get_tsc();
    t_ptr = &t;
    lsm = logsoftmax(x);
    for (idx_t b = 0; b < B; b++) {
      y(b) = -lsm(b,t(b));
    }
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("SoftmaxCrossEntropy<maxB=%ld,nC=%ld>.forward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)nC, (long)B, t1.ns - t0.ns);
    }
    return y;
  }
  
  array4<maxB,nC,1,1>& backward(vec<maxB>& gy) {
    if (opt.verbose>=2) {
      printf("SoftmaxCrossEntropy<maxB=%ld,nC=%ld>.backward(B=%ld)\n",
             (long)maxB, (long)nC, (long)B);
    }
    tsc_t t0 = get_tsc();
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
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("SoftmaxCrossEntropy<maxB=%ld,nC=%ld>.backward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)nC, (long)B, t1.ns - t0.ns);
    }
    return gx;
  }
};

int softmaxcrossentropy_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = opt.batch_sz;
  const idx_t nC = 10;
  rnd_gen_t rg;
  rg.seed(opt.weight_seed);
  SoftmaxCrossEntropy<maxB,nC> smxe;
  array4<maxB,nC,1,1> x;
  ivec<maxB> t;
  smxe.init(opt, B);
  x.init_uniform(B, rg, 0.0, 1.0);
  t.init_uniform(B, rg, 0, nC);
  smxe.forward(x, t);
  return 0;
}

