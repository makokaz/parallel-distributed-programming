/**
   @file maxpooling.h
 */
#pragma once

#include "vgg_util.h"
#include "vgg_arrays.h"

template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t S>
struct MaxPooling2D {
  cmdline_opt opt;
  idx_t B;
  array4<maxB,C,H/S,W/S> y;
  array4<maxB,C,H/S,W/S> max_idx;
  array4<maxB,C,H,W> gx;
  void init(cmdline_opt opt_, idx_t B_) {
    opt = opt_;
    B = B_;
    assert(B <= maxB);
    y.init(B);
    max_idx.init(B);
    gx.init(B);
  }
  array4<maxB,C,H/S,W/S>& forward(array4<maxB,C,H,W>& x) {
    if (opt.verbose>=2) {
      printf("Relu<maxB=%ld,C=%ld,H=%ld,W=%ld,S=%ld>.forward(B=%ld) starts\n",
             (long)maxB, (long)C, (long)H, (long)W, (long)S, (long)B);
    }
    tsc_t t0 = get_tsc();
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
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("Relu<maxB=%ld,C=%ld,H=%ld,W=%ld,S=%ld>.forward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)C, (long)H, (long)W, (long)S, (long)B, t1.ns - t0.ns);
    }
    return y;
  }
  array4<maxB,C,H,W>& backward(array4<maxB,C,H/S,W/S>& gy) {
    if (opt.verbose>=2) {
      printf("Relu<maxB=%ld,C=%ld,H=%ld,W=%ld,S=%ld>.backward(B=%ld) starts\n",
             (long)maxB, (long)C, (long)H, (long)W, (long)S, (long)B);
    }
    tsc_t t0 = get_tsc();
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
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("Relu<maxB=%ld,C=%ld,H=%ld,W=%ld,S=%ld>.backward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)C, (long)H, (long)W, (long)S, (long)B, t1.ns - t0.ns);
    }
    return gx;
  }
};

int maxpooling_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = opt.batch_sz;
  const idx_t C = 64;
  const idx_t H = 32;
  const idx_t W = 32;
  const idx_t S = 2;
  rnd_gen_t rg;
  rg.seed(opt.weight_seed);
  MaxPooling2D<maxB,C,H,W,S> mp;
  array4<maxB,C,H,W> x;
  mp.init(opt, B);
  x.init_uniform(B, rg, 0.0, 1.0);
  mp.forward(x);
  return 0;
}

