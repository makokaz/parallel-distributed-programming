/**
   @file block.h
 */
#pragma once

#include "vgg_util.h"
#include "vgg_arrays.h"
#include "convolution.h"
#include "batchnormalization.h"
#include "relu.h"

/** 
    batch size, input channel, kernel size, image_width, 
    image_height, out channel
*/
template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
struct Block {
  cmdline_opt opt;
  idx_t B;
  Convolution2D     <maxB,IC,H,W,K,OC> conv;
  BatchNormalization<maxB,OC,H,W>      bn;
  Relu              <maxB,OC,H,W>      relu;
  void init(cmdline_opt opt_, idx_t B_, rnd_gen_t& rg) {
    opt = opt_;
    B = B_;
    assert(B <= maxB);
    conv.init(opt, B, rg);
    bn.init(opt, B, rg);
    relu.init(opt, B);
  }
  void update(real eta) {
    conv.update(eta);
    bn.update(eta);
  }
  array4<maxB,OC,H,W>& forward(array4<maxB,IC,H,W>& x) {
    array4<maxB,OC,H,W>& x1 = conv.forward(x);
    array4<maxB,OC,H,W>& x2 = bn.forward(x1);
    array4<maxB,OC,H,W>&  y = relu.forward(x2);
    return y;
  }
  array4<maxB,IC,H,W>& backward(array4<maxB,OC,H,W>& gy) {
    array4<maxB,OC,H,W>& g2 = relu.backward(gy);
    array4<maxB,OC,H,W>& g1 = bn.backward(g2);
    array4<maxB,IC,H,W>& gx = conv.backward(g1);
    return gx;
  }
  void show_size() {
    printf(" conv :                 %10ld bytes\n", sizeof(conv));
    conv.show_size();
    printf(" bn   :                 %10ld bytes\n", sizeof(bn));
    printf(" relu :                 %10ld bytes\n", sizeof(relu));
  }
};

int block_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = opt.batch_sz;
  const idx_t IC = 3;
  const idx_t H = 32;
  const idx_t W = 32;
  const idx_t K = 1;
  const idx_t OC = 64;
  rnd_gen_t rg;
  rg.seed(opt.weight_seed);
  Block<maxB,IC,H,W,K,OC> block;
  array4<maxB,IC,H,W> x;
  block.init(opt, B, rg);
  x.init_uniform(B, rg, 0.0, 1.0);
  block.forward(x);
  return 0;
}

