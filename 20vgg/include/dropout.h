/**
   @file dropout.h
 */
#pragma once

#include "vgg_util.h"
#include "vgg_arrays.h"

template<idx_t maxB,idx_t C,idx_t H,idx_t W>
struct Dropout {
  cmdline_opt opt;
  rnd_gen_t rg;
  idx_t B;
  array4<maxB,C,H,W> y;
  array4<maxB,C,H,W> gx;
  real drop_ratio;
  long state_forward;
  void init(cmdline_opt opt_, idx_t B_, real drop_ratio_, long seed) {
    opt = opt_;
    B = B_;
    assert(B <= maxB);
    drop_ratio = drop_ratio_;
    rg.seed(seed);
    y.init(B);
    gx.init(B);
  }
  array4<maxB,C,W,H>& forward(array4<maxB,C,H,W>& x) {
    if (opt.verbose>=2) {
      printf("Dropout<maxB=%ld,C=%ld,H=%ld,W=%ld>.forward(B=%ld) starts\n",
             (long)maxB, (long)C, (long)H, (long)W, (long)B);
    }
    tsc_t t0 = get_tsc();
    /* zero elements with probability of ratio and
       scale others by 1/(1-ratio) so that the sum 
       will stay approximately the same */
    state_forward = rg.get_state();
    real scale = 1.0 / (1 - drop_ratio);
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            if (rg.rand01() < drop_ratio) {
              y(b,c,i,j) = 0.0;
            } else {
              y(b,c,i,j) = x(b,c,i,j) * scale;
            }
          }
        }
      }
    }
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("Dropout<maxB=%ld,C=%ld,H=%ld,W=%ld>.forward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)C, (long)H, (long)W, (long)B, t1.ns - t0.ns);
    }
    return y;
  }
  array4<maxB,C,H,W>& backward(array4<maxB,C,H,W>& gy) {
    if (opt.verbose>=2) {
      printf("Dropout<maxB=%ld,C=%ld,H=%ld,W=%ld>.backward(B=%ld) starts\n",
             (long)maxB, (long)C, (long)H, (long)W, (long)B);
    }
    tsc_t t0 = get_tsc();
    rg.seed(state_forward);
    real scale = 1.0 / (1 - drop_ratio);
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            if (rg.rand01() < drop_ratio) {
              gx(b,c,i,j) = 0.0;
            } else {
              gx(b,c,i,j) = scale * gy(b,c,i,j);
            }
          }
        }
      }
    }
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("Dropout<maxB=%ld,C=%ld,H=%ld,W=%ld>.backward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)C, (long)H, (long)W, (long)B, t1.ns - t0.ns);
    }
    return gx;
  }
};

int dropout_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  rnd_gen_t rg;
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = opt.batch_sz;
  const idx_t C = 64;
  const idx_t H = 32;
  const idx_t W = 32;
  const real drop_ratio = 0.3;
  rg.seed(opt.weight_seed);
  Dropout<maxB,C,H,W> dropout;
  array4<maxB,C,H,W> x;
  dropout.init(opt, B, drop_ratio, opt.dropout_seed);
  x.init_uniform(B, rg, 0.0, 1.0);
  dropout.forward(x);
  return 0;
}

