/**
   @file vgg.cc --- a single file implemention of VGG
 */

#include "include/vgg_util.h"
#include "include/vgg.h"
#include "include/cifar.h"

int main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  if (opt.error || opt.help) usage(argv[0]);
  const idx_t maxB = MAX_BATCH_SIZE;  /**< max batch size (constant) */
  const idx_t B  = opt.batch_sz;      /**< true batch size (<= maxB) */
  const idx_t C0 = 3;                  /**< input channels (RGB) */
  const idx_t H  = 32;                 /**< height of an image */
  const idx_t W  = 32;                 /**< width of an image */
  const idx_t K  = 1;        /**< kernel size (K=1 -> 3x3) */
  const idx_t S  = 2;        /**< max pooling stride HxW -> H/SxW/S */
  const idx_t C1 = 64;       /**< channels of the last stage */
  const idx_t nC = 10;       /**< number of classes */
  assert(B <= maxB);
  /* random number */
  rnd_gen_t rg;
  rg.seed(opt.weight_seed);
  /* build model and initialize weights */
  VGG<maxB,C0,H,W,K,S,C1,nC> * vgg = new VGG<maxB,C0,H,W,K,S,C1,nC>();
  vgg->init(opt, B, rg);
  /* data */
  cifar10_dataset<maxB,C0,H,W> data;
  data.load(opt.cifar_data, opt.start_data, opt.end_data);
  data.set_seed(opt.sample_seed);
  array4<maxB,C0,H,W> x;
  ivec<maxB> t;
  x.init(B);
  t.init(B);
  real eta = opt.learnrate;
  for (long i = 0; i < opt.iters; i++) {
    if (opt.verbose>=1) {
      printf("=== iter = %ld ===\n", i);
    }
    data.get_data_train(x, t, B);
    vgg->forward_backward_update(x, t, eta);
  }
  return 0;
}

