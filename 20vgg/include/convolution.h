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
    for (idx_t oc = 0; oc < OC; oc++) { // output channel
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
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            real s = 0.0;
            for (idx_t oc = 0; oc < OC; oc++) { // output channels
              for (idx_t i_ = max_i(-K,i-H+1); i_ <= min_i(K,i); i_++) {
                for (idx_t j_ = max_i(-K,j-W+1); j_ <= min_i(K,j); j_++) {
                  /* max(-K,i-H+1) <= i_ <= min(K,i)
                     i-H+1 <= i_ <= i
                     -i+H-1 >= -i_ >= -i
                     H-1 >= i-i_ >= 0
                  */
                  s += gy(b,oc,i-i_,j-j_) * w(oc,ic,i_,j_);
                }
              }
            }
            gx(b,ic,i,j) = s;
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

idx_t make_between(idx_t a, idx_t b, idx_t x) {
  return max_i(a, min_i(b - 1, x));
}

template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
  real convolution_grad_check_gx_1(Convolution2D<maxB,IC,H,W,K,OC>& conv,
                                   array4<maxB,IC,H,W> x,
                                   idx_t B,
                                   int iter, rnd_gen_t& rg) {
  const idx_t b  = rg.randi(0, B);
  const idx_t ic = rg.randi(0, IC);
  const idx_t oc = rg.randi(0, OC);
  const idx_t i  = rg.randi(0, H);
  const idx_t j  = rg.randi(0, W);
  const idx_t i_ = make_between(0, H, i + rg.randi(-K, K + 1));
  const idx_t j_ = make_between(0, W, j + rg.randi(-K, K + 1));
  const real e = 1.0e-3;
  /* make x - e and x + e */
  array4<maxB,IC,H,W> x_minus_e = x;
  array4<maxB,IC,H,W> x_plus_e = x;
  x_minus_e(b,ic,i,j) -= e;
  x_plus_e(b,ic,i,j)  += e;
  /* make y(x), y(x-e), y(x+e) */
  array4<maxB,OC,H,W> y_minus_e = conv.forward(x_minus_e);
  array4<maxB,OC,H,W> y_plus_e  = conv.forward(x_plus_e);
  /* gradient of y(b,oc,i_,j_) and back propagate
     -> dy(b,oc,i_,j_)/dx(*,*,*,*) */
  array4<maxB,OC,H,W> y = conv.forward(x);
  (void)y;
  array4<maxB,OC,H,W> gy;
  gy.init_const(B, 0.0);
  gy(b,oc,i_,j_) = 1.0;
  array4<maxB,IC,H,W> gx = conv.backward(gy);
  /* compare (y(x+e)-y(x-e))/2e and computed gradient */
  {
    real A = (y_plus_e(b,ic,i_,j_) - y_minus_e(b,ic,i_,j_)) / (2 * e);
    real B = gx(b,ic,i,j);
    
    real _A_ = fabs(A);
    real _B_ = fabs(B);
    real rel_e = (max_r(_A_, _B_) == 0 ? 0.0 : fabs(_A_ - _B_) / max_r(_A_, _B_));
    
    printf("%3d: A = (y(x+e)-y(x-e))/2e = %f\n", iter, A);
    printf("%3d: B = dy(%d,%d,%d,%d)/dx(%d,%d,%d,%d) = %f\n",
           iter, b,oc,i_,j_,b,ic,i,j,B);
    printf("%3d: relative diff = |A - B|/max(|A|,|B|) = %f\n", iter, rel_e);
    return rel_e;
  }
}

/**
   
   @brief check ∂y(b,oc,i,j)/∂w(oc,ic,i_,j_)

  */

template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
  real convolution_grad_check_gw_1(Convolution2D<maxB,IC,H,W,K,OC>& conv,
                                   array4<maxB,IC,H,W> x,
                                   idx_t B, int iter, rnd_gen_t& rg) {
  const idx_t b  = rg.randi(0, B);
  const idx_t ic = rg.randi(0, IC);
  const idx_t oc = rg.randi(0, OC);
  const idx_t i  = rg.randi(0, H);
  const idx_t j  = rg.randi(0, W);
  const idx_t i_ = rg.randi(-K, K + 1);
  const idx_t j_ = rg.randi(-K, K + 1);
  const real e = 1.0e-3;
  /* make x - e and x + e */
  Convolution2D<maxB,IC,H,W,K,OC> conv_minus_e = conv;
  Convolution2D<maxB,IC,H,W,K,OC> conv_plus_e = conv;
  conv_minus_e.w(oc,ic,i_,j_) -= e;
  conv_plus_e.w(oc,ic,i_,j_) += e;
  /* make y(x), y(x-e), y(x+e) */
  array4<maxB,OC,H,W> y_minus_e = conv_minus_e.forward(x);
  array4<maxB,OC,H,W> y_plus_e  = conv_plus_e.forward(x);
  /* gradient of y(b,oc,i_,j_) and back propagate
     -> dy(b,oc,i_,j_)/dx(*,*,*,*) */
  array4<maxB,OC,H,W> y = conv.forward(x);
  (void)y;
  array4<maxB,OC,H,W> gy;
  gy.init_const(B, 0.0);
  gy(b,oc,i,j) = 1.0;
  array4<maxB,IC,H,W> gx = conv.backward(gy);
  (void)gx;
  
  /* compare (y(x+e)-y(x-e))/2e and computed gradient */
  {
    real A = (y_plus_e(b,ic,i,j) - y_minus_e(b,ic,i,j)) / (2 * e);
    real B = conv.gw(oc,ic,i_,j_);
    real _A_ = fabs(A);
    real _B_ = fabs(B);
    real rel_e = (max_r(_A_, _B_) == 0 ? 0.0 : fabs(_A_ - _B_) / max_r(_A_, _B_));
    
    printf("%3d: A = (y(x+e)-y(x-e))/2e = %f\n", iter, A);
    printf("%3d: B = dy(%d,%d,%d,%d)/dw(%d,%d,%d,%d) = %f\n",
           iter, b,oc,i,j,oc,ic,i_,j_,B);
    printf("%3d: relative diff = |A - B|/max(|A|,|B|) = %f\n", iter, rel_e);
    return rel_e;
  }
}


int convolution_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = 1;
  const idx_t B = min_i(maxB, opt.batch_sz);
  const idx_t IC = 1;
  const idx_t H = 3;
  const idx_t W = 3;
  const idx_t K = 1;
  const idx_t OC = 1;
  const int n_checks_gx = 0;
  const int n_checks_gw = 10;

  /* initialize random number generator */
  rnd_gen_t rg;
  rg.seed(opt.weight_seed);
  /* initialize convolution parameters */
  Convolution2D<maxB,IC,H,W,K,OC> conv;
  conv.init(opt, B, rg);
  /* initialize input x  */
  array4<maxB,IC,H,W> x;
  x.init_uniform(B, rg, 0.0, 1.0);

  if (n_checks_gx > 0) {
    real max_rel_e_gx = 0.0;
    for (int t = 0; t < n_checks_gx; t++) {
      real rel_e = convolution_grad_check_gx_1(conv, x, B, t, rg);
      max_rel_e_gx = max_r(max_rel_e_gx, rel_e);
    }
    printf("max relative error for gx = %f\n", max_rel_e_gx);
  }

  if (n_checks_gw > 0) {
    real max_rel_e_gw = 0.0;
    for (int t = 0; t < n_checks_gw; t++) {
      real rel_e = convolution_grad_check_gw_1(conv, x, B, t, rg);
      max_rel_e_gw = max_r(max_rel_e_gw, rel_e);
    }
    printf("max relative error for gw = %f\n", max_rel_e_gw);
  }
  
  return 0;
}
