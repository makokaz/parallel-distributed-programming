/**
   @file vgg.cc --- a C++ implemention of VGG
 */

#include "include/vgg_util.h"
#include "include/vgg.h"
#include "include/cifar.h"

/**
   @brief grab a mini batch (B training samples), forward, backward and update.
   @return the average loss of the mini batch.
 */
template<idx_t maxB,idx_t C0,idx_t H,idx_t W,idx_t K,idx_t S,idx_t C1,idx_t nC>
static real train(VGG<maxB,C0,H,W,K,S,C1,nC> * vgg,
                  cifar10_dataset<maxB,C0,H,W>& data, idx_t B, long count) {
  vgg->lgr->log(1, "=== train %ld - %ld ===", count, count + B);
  if (vgg->opt.single_batch) {
    data.set_seed(vgg->opt.sample_seed);
  }
  data.get_data_train(vgg->x, vgg->t, B);
  real Lsum = vgg->forward_backward_update(vgg->x, vgg->t, vgg->opt.learnrate);
  real L = Lsum / B;
  vgg->lgr->log(1, "train loss = %.9f", L);
  return L;
}

/**
   @brief forward compute B_validate validation samples 
   (taking several mini batches if necessary)
   @return the average loss of the validation data
 */
template<idx_t maxB,idx_t C0,idx_t H,idx_t W,idx_t K,idx_t S,idx_t C1,idx_t nC>
static real validate(VGG<maxB,C0,H,W,K,S,C1,nC> * vgg,
                     cifar10_dataset<maxB,C0,H,W>& data, long count) {
  if (data.n_validate > 0) {
    vgg->lgr->log(1, "=== validate %ld - %ld ===", count, count + data.n_validate);
    long read_from = 0;
    real Lsum = 0.0;
    while (read_from < data.n_validate) {
      long read_to = min_i(read_from + maxB, data.n_validate);
      data.get_data_validate(vgg->x, vgg->t, read_from, read_to);
      vec<maxB>& y = vgg->forward(vgg->x, vgg->t);
      y.to_host();
      Lsum += y.sum();
      read_from = read_to; 
    }
    real L = Lsum / data.n_validate;
    vgg->lgr->log(1, "validate loss = %.9f", L);
    return L;
  } else {
    return 0.0;
  }
}

/**
   @brief main function of VGG
   @details Train VGG network with data from the file specified by
   --cifar_data/-d (default: cifar-10-batches-bin/data_batch_1.bin).
   If you want to use only a part of data, you can specify a range
   by --start_data and --end_data. e.g., --start
   Take a number of samples specified by --batch_sz/-d
   samples at a time for training.
   occasionally (every 
   hold out 
   @return the average loss of the validation data
 */
int main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  if (opt.error || opt.help) usage(argv[0]);
  const idx_t maxB = MAX_BATCH_SIZE;  /**< max batch size (constant) */
  const idx_t B  = opt.batch_sz;      /**< true batch size (<= maxB) */
  const idx_t C0 = 3;                 /**< input channels (RGB) */
  const idx_t H  = 32;                /**< height of an image */
  const idx_t W  = 32;                /**< width of an image */
  const idx_t K  = 1;        /**< kernel size (K=1 -> 3x3) */
  const idx_t S  = 2;        /**< max pooling stride HxW -> H/SxW/S */
  const idx_t C1 = 64;       /**< channels of the last stage */
  const idx_t nC = 10;       /**< number of classes */
  assert(B <= maxB);
  /* logger */
  logger lgr;
  lgr.start_log(opt);
  /* random number */
  rnd_gen_t rg;
  rg.seed(opt.weight_seed);
  /* build model and initialize weights */
  lgr.log(1, "model building starts");
  VGG<maxB,C0,H,W,K,S,C1,nC> * vgg = new VGG<maxB,C0,H,W,K,S,C1,nC>();
  vgg->init(opt, &lgr, rg);
  vgg->make_dev();
  vgg->to_dev();
  lgr.log(1, "model building ends");
  /* data */
  lgr.log(1, "data loading starts");
  cifar10_dataset<maxB,C0,H,W> data;
  data.load(opt.cifar_data, opt.start_data, opt.end_data,
            opt.validate_ratio, opt.validate_seed);
  data.set_seed(opt.sample_seed);
  lgr.log(1, "data loading ends");
  long n_trained = 0;
  long n_validated = 0;
  lgr.log(1, "training starts");
  for (long i = 0; i < opt.iters; i++) {
    real train_loss = train(vgg, data, B, n_trained);
    (void)train_loss;
    n_trained += B;
    if (data.n_validate > 0 &&
        n_trained >= opt.validate_interval * (n_validated + data.n_validate)) {
      real validate_loss = validate(vgg, data, n_validated);
      (void)validate_loss;
      n_validated += data.n_validate;
    }
  }
  lgr.log(1, "training ends");
  lgr.end_log();
  return 0;
}

