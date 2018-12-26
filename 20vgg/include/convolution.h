/**
   @file convolution.h
 */
#pragma once

#include "vgg_util.h"
#include "vgg_arrays.h"

/**
   @brief convolution to images

   @param maxB : the number of images (batch size)
   @param IC : the number of channels per input image (the 
               original input has typically three channels for RGB. 
               in hidden layers, it starts from 64 and goes up 
               to 512 in the last hidden layer)
   @param W : width of an image (32 for an input image, down to 1 in
              the last hidden layer)
   @param H : height of an image
   @param K : the kernel size (KxK pixels) that applies to each pixel.
   @param OC : the number of channels per an output image

   @details : this layer essentially converts each ICxWxH image
              to OCxWxH image, applying ICxKxK stencil to each pixel

 */
template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
struct Convolution2D {
  cmdline_opt opt;
  idx_t B;
  array4<maxB,IC,H,W>* x_ptr;      /**< pointer to input */
  warray4<OC,IC,K,K> w;            /**< weight of the filter */ 
  array4<maxB,OC,H,W> y;           /**< output */
  warray4<OC,IC,K,K> gw;           /**< gradient of loss wrt w */
  array4<maxB,IC,H,W> gx;          /**< gradient of loss wrt x */
  void init(cmdline_opt opt_, idx_t B_, rnd_gen_t& rg) {
    opt = opt_;
    B = B_;
    assert(B <= maxB);
    w.init_normal(rg, 0.0, 1 / sqrt((2 * K + 1) * (2 * K + 1) * IC));
    y.init(B);
    gx.init(B);
  }
  void update(real eta) {
    w.update(eta, gw);
  }
  array4<maxB,OC,H,W>& forward(array4<maxB,IC,H,W>& x) {
    if (opt.verbose>=2) {
      printf("Convolution2D<maxB=%ld,IC=%ld,H=%ld,W=%ld,K=%ld,OC=%ld>.forward(B=%ld) starts\n",
             (long)maxB, (long)IC, (long)H, (long)W, (long)K, (long)OC, (long)B);
    }
    tsc_t t0 = get_tsc();
    x_ptr = &x;
    for (idx_t b = 0; b < B; b++) { // samples
      for (idx_t oc = 0; oc < OC; oc++) { // output channels
        for (idx_t i = 0; i < H; i++) {   // width
          for (idx_t j = 0; j < W; j++) { // height
            real s = 0.0;
            for (idx_t ic = 0; ic < IC; ic++) { // input channel
              /* -K<=i_<=K, 0<=i+i_<H => -K<=i_&-i<i_; i_<=K&i_<H-i*/
              for (idx_t i_ = max_i(-K,-i); i_ <= min_i(K,H-i-1); i_++) {
                for (idx_t j_ = max_i(-K,-j); j_ <= min_i(K,W-j-1); j_++) {
                  s += w(oc,ic,i_,j_) * x(b,ic,i+i_,j+j_);
                }
              }
            }
            y(b,oc,i,j) = s;
          }
        }
      }
    }
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("Convolution2D<maxB=%ld,IC=%ld,H=%ld,W=%ld,K=%ld,OC=%ld>.forward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)IC, (long)H, (long)W, (long)K, (long)OC,
             (long)B, t1.ns - t0.ns);
    }
    return y;
  }
  array4<maxB,IC,H,W>& backward(array4<maxB,OC,H,W>& gy) {
    if (opt.verbose>=2) {
      printf("Convolution2D<maxB=%ld,IC=%ld,H=%ld,W=%ld,K=%ld,OC=%ld>.backward(B=%ld) starts\n",
             (long)maxB, (long)IC, (long)H, (long)W, (long)K, (long)OC, (long)B);
    }
    tsc_t t0 = get_tsc();
    array4<maxB,IC,H,W>& x = *x_ptr;
    for (idx_t oc = 0; oc < OC; oc++) { // samples
      for (idx_t ic = 0; ic < IC; ic++) { // input channel
        for (idx_t i_ = -K; i_ <= K; i_++) {
          for (idx_t j_ = -K; j_ <= K; j_++) {
            real s = 0.0;
            for (idx_t b = 0; b < B; b++) { // samples
              for (idx_t i = max_i(0,-i_); i < min_i(H,H-i_); i++) {   // width
                for (idx_t j = max_i(0,-j_); j < min_i(W,W-j_); j++) { // height
                  s += gy(b,oc,i,j) * x(b,ic,i+i_,j+j_);
                }
              }
            }
            gw(oc,ic,i_,j_) = s;
          }
        }
      }
    }
    for (idx_t b = 0; b < B; b++) { // samples
      for (idx_t ic = 0; ic < IC; ic++) { // input channel
        for (idx_t i_ = -K; i_ <= K; i_++) {
          for (idx_t j_ = -K; j_ <= K; j_++) {
            /* 0<=i<H, 0<=i+i_<H => 0<=i&-i_<=i; i<H&i<H-i_ */
            real s = 0.0;
            for (idx_t i = max_i(0,-i_); i < min_i(H,H-i_); i++) {   // width
              for (idx_t j = max_i(0,-j_); j < min_i(W,W-j_); j++) { // height
                for (idx_t oc = 0; oc < OC; oc++) { // output channels
                  s += gy(b,oc,i,j) * w(oc,ic,i_,j_);
                }
              }
            }
            // need to reindex
            //gx(b,ic,i+i_,j+j_) = s;
          }
        }
      }
    }
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) {
      printf("Convolution2D<maxB=%ld,IC=%ld,H=%ld,W=%ld,K=%ld,OC=%ld>.backward(B=%ld) ends in %ld ns\n",
             (long)maxB, (long)IC, (long)H, (long)W, (long)K, (long)OC,
             (long)B, t1.ns - t0.ns);
    }
    return gx;
  }
  void show_size() {
    printf("  w  : %d x %d x %d x %d = %10ld bytes\n",
           OC, IC, (2*K+1), (2*K+1), sizeof(w));
    printf("  y  :                  %10ld bytes\n", sizeof(y));
    printf("  gw :                  %10ld bytes\n", sizeof(gw));
    printf("  gx :                  %10ld bytes\n", sizeof(gw));
  }
};

int convolution_main(int argc, char ** argv) {
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
  Convolution2D<maxB,IC,H,W,K,OC> conv;
  array4<maxB,IC,H,W> x;
  conv.init(opt, B, rg);
  x.init_uniform(B, rg, 0.0, 1.0);
  conv.forward(x);
  return 0;
}
