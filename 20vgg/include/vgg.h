/**
   @file vgg.h
 */
#pragma once

#include "vgg_util.h"
#include "vgg_arrays.h"
#include "block.h"
#include "linear.h"
#include "dropout.h"
#include "maxpooling.h"
#include "softmaxcrossentropy.h"

/**
   maxB : batch size (64)
   iC : number of channels in an input image (3)
   hC : number of channels in the first hidden layer (64)
        (those in the following layers are hC/2, hC/4, hC/8, ...)
   K : convolution kernel size (3x3)
   W : width of an input image (32)
   H : height of an input image (32)
   nC : number of classes (10)
 */
template<idx_t maxB,idx_t C0,idx_t H,idx_t W,idx_t K,idx_t S,idx_t C1,idx_t nC>
struct VGG {
  cmdline_opt opt;
  idx_t B;
  /* group 1 : 64 channels x 32x32 */
  static const idx_t H1 = H, W1 = W;
  Block       <maxB,C0,H1,W1,K,C1> block1_1;
  Dropout     <maxB,C1,H1,W1>      dropout1_1;
  Block       <maxB,C1,H1,W1,K,C1> block1_2;
  MaxPooling2D<maxB,C1,H1,W1,S>    max_pooling_2d1;

  static const idx_t H2 = H/S, W2 = W/S, C2 = S * C1;
  /* group 2 : 64 channels x 16x16 */
  Block       <maxB,C1,H2,W2,K,C2> block2_1;
  Dropout     <maxB,C2,H2,W2>      dropout2_1;
  Block       <maxB,C2,H2,W2,K,C2> block2_2;
  MaxPooling2D<maxB,C2,H2,W2,S>    max_pooling_2d2;

  static const idx_t H3 = H/(S*S), W3 = W/(S*S), C3 = S*S * C1;
  /* group 3 : 128 channels x 8x8 */
  Block       <maxB,C2,H3,W3,K,C3> block3_1;
  Dropout     <maxB,C3,H3,W3>      dropout3_1;
  Block       <maxB,C3,H3,W3,K,C3> block3_2;
  Dropout     <maxB,C3,H3,W3>      dropout3_2;
  Block       <maxB,C3,H3,W3,K,C3> block3_3;
  MaxPooling2D<maxB,C3,H3,W3,S>    max_pooling_2d3;
  
  static const idx_t H4 = H/(S*S*S), W4 = W/(S*S*S), C4 = S*S*S * C1;
  /* group 4 : 256 channels x 4x4 */
  Block       <maxB,C3,H4,W4,K,C4> block4_1;
  Dropout     <maxB,C4,H4,W4>      dropout4_1;
  Block       <maxB,C4,H4,W4,K,C4> block4_2;
  Dropout     <maxB,C4,H4,W4>      dropout4_2;
  Block       <maxB,C4,H4,W4,K,C4> block4_3;
  MaxPooling2D<maxB,C4,H4,W4,S>    max_pooling_2d4;
  
  static const idx_t H5 = H/(S*S*S*S), W5 = W/(S*S*S*S);
  /* group 5 : 512 channels x 2x2 */
  Block       <maxB,C4,H5,W5,K,C4> block5_1;
  Dropout     <maxB,C4,H5,W5>      dropout5_1;
  Block       <maxB,C4,H5,W5,K,C4> block5_2;
  Dropout     <maxB,C4,H5,W5>      dropout5_2;
  Block       <maxB,C4,H5,W5,K,C4> block5_3;
  MaxPooling2D<maxB,C4,H5,W5,S>    max_pooling_2d5;
  
  /* group 6 : 512 channels x 1x1 */
  static const idx_t H6 = H/(S*S*S*S*S), W6 = W/(S*S*S*S*S);
  Dropout           <maxB,C4,H6,W6> dropout6_1;
  Linear            <maxB,C4,C4>    fc1;
  BatchNormalization<maxB,C4,H6,W6> bn_fc1;
  Relu              <maxB,C4,H6,W6> relu;
  Dropout           <maxB,C4,H6,W6> dropout6_2;
  Linear            <maxB,C4,nC>    fc2;

  SoftmaxCrossEntropy<maxB,nC>       softmax_cross_entropy;

  vec<maxB>              loss;
  /** 
      @brief forward
  */
  vec<maxB>& forward(array4<maxB,C0,H,W>& x, ivec<maxB>& t) {
    /* 64 channel blocks */
    array4<maxB,C1,H1,W1>&  x1 = block1_1.forward(x);
    array4<maxB,C1,H1,W1>&  x2 = dropout1_1.forward(x1);
    array4<maxB,C1,H1,W1>&  x3 = block1_2.forward(x2);
    array4<maxB,C1,H2,W2>&  x4 = max_pooling_2d1.forward(x3);
    /* 128 channel blocks */
    array4<maxB,C2,H2,W2>&  x5 = block2_1.forward(x4);
    array4<maxB,C2,H2,W2>&  x6 = dropout2_1.forward(x5);
    array4<maxB,C2,H2,W2>&  x7 = block2_2.forward(x6);
    array4<maxB,C2,H3,W3>&  x8 = max_pooling_2d2.forward(x7);
    /* 256 channel blocks */
    array4<maxB,C3,H3,W3>&  x9 = block3_1.forward(x8);
    array4<maxB,C3,H3,W3>& x10 = dropout3_1.forward(x9);
    array4<maxB,C3,H3,W3>& x11 = block3_2.forward(x10);
    array4<maxB,C3,H3,W3>& x12 = dropout3_2.forward(x11);
    array4<maxB,C3,H3,W3>& x13 = block3_3.forward(x12);
    array4<maxB,C3,H4,W4>& x14 = max_pooling_2d3.forward(x13);
    /* 512 channel blocks */
    array4<maxB,C4,H4,W4>& x15 = block4_1.forward(x14);
    array4<maxB,C4,H4,W4>& x16 = dropout4_1.forward(x15);
    array4<maxB,C4,H4,W4>& x17 = block4_2.forward(x16);
    array4<maxB,C4,H4,W4>& x18 = dropout4_2.forward(x17);
    array4<maxB,C4,H4,W4>& x19 = block4_3.forward(x18);
    array4<maxB,C4,H5,W5>& x20 = max_pooling_2d4.forward(x19);
    /* 512 channel blocks */
    array4<maxB,C4,H5,W5>& x21 = block5_1.forward(x20);
    array4<maxB,C4,H5,W5>& x22 = dropout5_1.forward(x21);
    array4<maxB,C4,H5,W5>& x23 = block5_2.forward(x22);
    array4<maxB,C4,H5,W5>& x24 = dropout5_2.forward(x23);
    array4<maxB,C4,H5,W5>& x25 = block5_3.forward(x24);
    array4<maxB,C4,H6,W6>& x26 = max_pooling_2d5.forward(x25);
    
    array4<maxB,C4,H6,W6>& x27 = dropout6_1.forward(x26);
    array4<maxB,C4,H6,W6>& x28 = fc1.forward(x27);
    array4<maxB,C4,H6,W6>& x29 = bn_fc1.forward(x28);
    array4<maxB,C4,H6,W6>& x30 = relu.forward(x29);
    array4<maxB,C4,H6,W6>& x31 = dropout6_2.forward(x30);
    array4<maxB,nC,H6,W6>& x32 = fc2.forward(x31);
    loss = softmax_cross_entropy.forward(x32, t);
    return loss;
  }

  /** 
      @brief backward
  */
  void backward(vec<maxB>& gloss) {
    array4<maxB,nC,H6,W6>& g32 = softmax_cross_entropy.backward(gloss);
    array4<maxB,C4,H6,W6>& g31 = fc2.backward(g32);
    array4<maxB,C4,H6,W6>& g30 = dropout6_2.backward(g31);
    array4<maxB,C4,H6,W6>& g29 = relu.backward(g30);
    array4<maxB,C4,H6,W6>& g28 = bn_fc1.backward(g29);
    array4<maxB,C4,H6,W6>& g27 = fc1.backward(g28);
    array4<maxB,C4,H6,W6>& g26 = dropout6_1.backward(g27);
    /* 512 channel blocks */
    array4<maxB,C4,H5,W5>& g25 = max_pooling_2d5.backward(g26);
    array4<maxB,C4,H5,W5>& g24 = block5_3.backward(g25);
    array4<maxB,C4,H5,W5>& g23 = dropout5_2.backward(g24);
    array4<maxB,C4,H5,W5>& g22 = block5_2.backward(g23);
    array4<maxB,C4,H5,W5>& g21 = dropout5_1.backward(g22);
    /* 512 channel blocks */
    array4<maxB,C4,H5,W5>& g20 = block5_1.backward(g21);
    array4<maxB,C4,H4,W4>& g19 = max_pooling_2d4.backward(g20);
    array4<maxB,C4,H4,W4>& g18 = block4_3.backward(g19);
    array4<maxB,C4,H4,W4>& g17 = dropout4_2.backward(g18);
    array4<maxB,C4,H4,W4>& g16 = block4_2.backward(g17);
    array4<maxB,C4,H4,W4>& g15 = dropout4_1.backward(g16);
    /* 256 channel blocks */
    array4<maxB,C3,H4,W4>& g14 = block4_1.backward(g15);
    array4<maxB,C3,H3,W3>& g13 = max_pooling_2d3.backward(g14);
    array4<maxB,C3,H3,W3>& g12 = block3_3.backward(g13);
    array4<maxB,C3,H3,W3>& g11 = dropout3_2.backward(g12);
    array4<maxB,C3,H3,W3>& g10 = block3_2.backward(g11);
    array4<maxB,C3,H3,W3>&  g9 = dropout3_1.backward(g10);
    /* 128 channel blocks */
    array4<maxB,C2,H3,W3>&  g8 = block3_1.backward(g9);
    array4<maxB,C2,H2,W2>&  g7 = max_pooling_2d2.backward(g8);
    array4<maxB,C2,H2,W2>&  g6 = block2_2.backward(g7);
    array4<maxB,C2,H2,W2>&  g5 = dropout2_1.backward(g6);
    /* 64 channel blocks */
    array4<maxB,C1,H2,W2>&  g4 = block2_1.backward(g5);
    array4<maxB,C1,H1,W1>&  g3 = max_pooling_2d1.backward(g4);
    array4<maxB,C1,H1,W1>&  g2 = block1_2.backward(g3);
    array4<maxB,C1,H1,W1>&  g1 = dropout1_1.backward(g2);
    array4<maxB,C0,H1,W1>&  g0 = block1_1.backward(g1);
    (void)g0;
  }

  void init(cmdline_opt opt_, idx_t B_, rnd_gen_t& rg) {
    opt = opt_;
    B = B_;
    assert(B <= maxB);

    long seed = opt.dropout_seed;

    block1_1.init(opt, B, rg);
    dropout1_1.init(opt, B, 0.3, seed += 100);
    block1_2.init(opt, B, rg);
    max_pooling_2d1.init(opt, B);

    block2_1.init(opt, B, rg);
    dropout2_1.init(opt, B, 0.4, seed += 100);
    block2_2.init(opt, B, rg);
    max_pooling_2d2.init(opt, B);

    block3_1.init(opt, B, rg);
    dropout3_1.init(opt, B, 0.4, seed += 100);
    block3_2.init(opt, B, rg);
    dropout3_2.init(opt, B, 0.4, seed += 100);
    block3_3.init(opt, B, rg);
    max_pooling_2d3.init(opt, B);

    block4_1.init(opt, B, rg);
    dropout4_1.init(opt, B, 0.4, seed += 100);
    block4_2.init(opt, B, rg);
    dropout4_2.init(opt, B, 0.4, seed += 100);
    block4_3.init(opt, B, rg);
    max_pooling_2d4.init(opt, B);

    block5_1.init(opt, B, rg);
    dropout5_1.init(opt, B, 0.4, seed += 100);
    block5_2.init(opt, B, rg);
    dropout5_2.init(opt, B, 0.4, seed += 100);
    block5_3.init(opt, B, rg);
    max_pooling_2d5.init(opt, B);

    dropout6_1.init(opt, B, 0.5, seed += 100);
    fc1.init(opt, B, rg);
    bn_fc1.init(opt, B, rg);
    relu.init(opt, B);
    dropout6_2.init(opt, B, 0.5, seed += 100);
    fc2.init(opt, B, rg);
    
    softmax_cross_entropy.init(opt, B);
    loss.init_const(B, 0);
  }

  void update(real eta) {
    block1_1.update(eta);
    block1_2.update(eta);

    block2_1.update(eta);
    block2_2.update(eta);

    block3_1.update(eta);
    block3_2.update(eta);
    block3_3.update(eta);
  
    block4_1.update(eta);
    block4_2.update(eta);
    block4_3.update(eta);
  
    block5_1.update(eta);
    block5_2.update(eta);
    block5_3.update(eta);
  
    fc1.update(eta);
    fc2.update(eta);
  }
  
  void forward_backward_update(array4<maxB,C0,H,W>& x, ivec<maxB>& t,
                               real eta) {
    vec<maxB>& y = forward(x, t);
    real L = y.sum();
    if (opt.verbose>=2) {
      printf("L = %f\n", L);
    }
    vec<maxB> gy;
    gy.init_const(B, 1.0);
    backward(gy);
    update(eta);
  }


  void show_size() {
    printf("data size\n");
    printf("VGG :                   %10ld bytes\n", sizeof(*this));
    printf("block1_1 :              %10ld bytes\n", sizeof(block1_1));
    block1_1.show_size();
    printf("dropout1_1 :            %10ld bytes\n", sizeof(dropout1_1));
    printf("block1_2 :              %10ld bytes\n", sizeof(block1_2));
    block1_2.show_size();
    printf("max_pooling_2d1 :       %10ld bytes\n", sizeof(max_pooling_2d1));
    printf("block2_1 :              %10ld bytes\n", sizeof(block2_1));
    block2_1.show_size();
    printf("dropout2_1 :            %10ld bytes\n", sizeof(dropout2_1));
    printf("block2_2 :              %10ld bytes\n", sizeof(block2_2));
    block2_2.show_size();
    printf("max_pooling_2d2 :       %10ld bytes\n", sizeof(max_pooling_2d2));
    printf("block3_1 :              %10ld bytes\n", sizeof(block3_1));
    block3_1.show_size();
    printf("dropout3_1 :            %10ld bytes\n", sizeof(dropout3_1));
    printf("block3_2 :              %10ld bytes\n", sizeof(block3_2));
    block3_2.show_size();
    printf("dropout3_2 :            %10ld bytes\n", sizeof(dropout3_2));
    printf("block3_3 :              %10ld bytes\n", sizeof(block3_3));
    block3_3.show_size();
    printf("max_pooling_2d3 :       %10ld bytes\n", sizeof(max_pooling_2d3));
    printf("block4_1 :              %10ld bytes\n", sizeof(block4_1));
    block4_1.show_size();
    printf("dropout4_1 :            %10ld bytes\n", sizeof(dropout4_1));
    printf("block4_2 :              %10ld bytes\n", sizeof(block4_2));
    block4_2.show_size();
    printf("dropout4_2 :            %10ld bytes\n", sizeof(dropout4_2));
    printf("block4_3 :              %10ld bytes\n", sizeof(block4_3));
    block4_3.show_size();
    printf("max_pooling_2d4 :       %10ld bytes\n", sizeof(max_pooling_2d4));
    printf("block5_1 :              %10ld bytes\n", sizeof(block5_1));
    block5_1.show_size();
    printf("dropout5_1 :            %10ld bytes\n", sizeof(dropout5_1));
    printf("block5_2 :              %10ld bytes\n", sizeof(block5_2));
    block5_2.show_size();
    printf("dropout5_2 :            %10ld bytes\n", sizeof(dropout5_2));
    printf("block5_3 :              %10ld bytes\n", sizeof(block5_3));
    block5_3.show_size();
    printf("max_pooling_2d5 :       %10ld bytes\n", sizeof(max_pooling_2d5));
    printf("dropout6_1 :            %10ld bytes\n", sizeof(dropout6_1));
    printf("fc1 :                   %10ld bytes\n", sizeof(fc1));
    printf("bn_fc1 :                %10ld bytes\n", sizeof(bn_fc1));
    printf("relu :                  %10ld bytes\n", sizeof(relu));
    printf("dropout6_2 :            %10ld bytes\n", sizeof(dropout6_2));
    printf("fc2 :                   %10ld bytes\n", sizeof(fc2));
    printf("softmax_cross_entropy : %10ld bytes\n", sizeof(softmax_cross_entropy));
    printf("loss :                  %10ld bytes\n", sizeof(loss));
  }
};

int vgg_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const int maxB = MAX_BATCH_SIZE;
  const int B = opt.batch_sz;
  const int C0 = 3;
  const int H = 32;
  const int W = 32;
  const int K = 1;
  const int S = 2;
  const int C1 = 512;
  const int nC = 10;
  rnd_gen_t rg;
  rg.seed(opt.sample_seed);
  VGG<maxB,C0,H,W,K,S,C1,nC> * vgg = new VGG<maxB,C0,H,W,K,S,C1,nC>();
  vgg->init(opt, B, rg);
  return 0;
}

