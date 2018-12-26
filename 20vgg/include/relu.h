/**
   @file relu.h
 */
#pragma once

#include "vgg_util.h"
#include "vgg_arrays.h"

template<idx_t maxB,idx_t C,idx_t H,idx_t W>
struct Relu {
  cmdline_opt opt;
  idx_t B;
  array4<maxB,C,H,W>* x_ptr;
  array4<maxB,C,H,W> y;
  array4<maxB,C,H,W> gx;
  void init(cmdline_opt opt_, idx_t B_) {
    opt = opt_;
    B = B_;
    assert(B <= maxB);
    y.init(B);
    gx.init(B);
  }
  array4<maxB,C,H,W>& forward(array4<maxB,C,H,W>& x) {
    if (opt.verbose>=2) {
      printf("Relu<maxB=%ld,C=%ld,H=%ld,W=%ld>.forward(B=%ld) starts\n",
             (long)maxB, (long)C, (long)H, (long)W, (long)B);
    }
    tsc_t t0 = get_tsc();
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
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("Relu<maxB=%ld,C=%ld,H=%ld,W=%ld>.forward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)C, (long)H, (long)W, (long)B, t1.ns - t0.ns);
    }
    return y;
  }
  array4<maxB,C,H,W>& backward(array4<maxB,C,H,W>& gy) {
    if (opt.verbose>=2) {
      printf("Relu<maxB=%ld,C=%ld,H=%ld,W=%ld>.backward(B=%ld)\n",
             (long)maxB, (long)C, (long)H, (long)W, (long)B);
    }
    tsc_t t0 = get_tsc();
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
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("Relu<maxB=%ld,C=%ld,H=%ld,W=%ld>.backward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)C, (long)H, (long)W, (long)B, t1.ns - t0.ns);
    }
    return gx;
  }
};

int relu_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = opt.batch_sz;
  const idx_t C = 64;
  const idx_t H = 32;
  const idx_t W = 32;
  rnd_gen_t rg;
  rg.seed(opt.weight_seed);
  Relu<maxB,C,H,W> relu;
  array4<maxB,C,H,W> x;
  relu.init(opt, B);
  x.init_uniform(B, rg, 0.0, 1.0);
  relu.forward(x);
  return 0;
}

