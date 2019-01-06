/**
   @file cifar.h
 */
#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "vgg_util.h"
#include "vgg_arrays.h"

template<idx_t IC,idx_t H,idx_t W>
struct cifar10_data_item {
  int index;
  real w[IC][H][W];
  char label;
  real& operator()(idx_t ic, idx_t i, idx_t j) {
    range_chk(0, ic, IC);
    range_chk(0, i, H);
    range_chk(0, j, W);
    return w[ic][i][j];
  }
};

template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
struct cifar10_dataset {
  long n_data;
  long n_validate;
  long n_train;
  cifar10_data_item<IC,H,W> * dataset; /**< whole dataset read (n_data items) */
  cifar10_data_item<IC,H,W> * validate; /**< for validation (first n_validate items) */
  cifar10_data_item<IC,H,W> * train;   /**< training part (remaining items) */
  rnd_gen_t rg;
  void set_seed(long sd) {
    rg.seed(sd);
  }
  long get_n_data_in_file(const char * cifar_bin) {
    struct stat sb[1];
    if (lstat(cifar_bin, sb) == -1) {
      perror("lstat");
      fprintf(stderr,
              "error: could not find the cifar data file %s. Specify a right file name with -d FILENAME or do this so that it can find the default file (cifar-10-batches-bin/data_batch_1.bin).\n"
              "\n"
              "$ ln -s /home/tau/cifar10/cifar-10-batches-bin\n",
              cifar_bin);
      exit(1);
    }
    long sz = sb->st_size;
    long sz1 = IC * H * W + 1;
    assert(sz % sz1 == 0);
    return sz / sz1;
  }
  int load(logger& lgr,
           const char * cifar_bin, long start, long end,
           double validate_ratio, long validate_seed) {
    long n_data_in_file = get_n_data_in_file(cifar_bin);
    if (end == 0) end = n_data_in_file;
    assert(start >= 0);
    assert(end <= n_data_in_file);
    assert(start < end);

    n_data = end - start;
    n_train = max_i(1, n_data - n_data * validate_ratio);
    n_validate = n_data - n_train;
    lgr.log(1, "loading data from %s [%ld:%ld] (%ld/%ld train/validate) starts",
            cifar_bin, start, end, n_train, n_validate);
    if (n_validate == 0) {
      printf("warning: no data left for validation (validation not performed)\n");
    }
    
    dataset = new cifar10_data_item<IC,H,W> [n_data];
    validate = dataset;
    train = dataset + n_validate;
    FILE * fp = fopen(cifar_bin, "rb");
    if (!fp) { perror("fopen"); exit(1); }
    if (fseek(fp, start * (1 + IC * H * W), SEEK_SET) == -1) {
      perror("fseek"); exit(1);
    }
    for (int k = 0; k < n_data; k++) {
      unsigned char label[1];
      unsigned char rgb[IC][H][W];
      size_t r = fread(label, sizeof(label), 1, fp);
      if (ferror(fp)) { perror("fread"); exit(1); }
      if (feof(fp)) break;
      r = fread(rgb, sizeof(rgb), 1, fp);
      if (r != 1) { perror("fread"); exit(1); }
      int max_value = 0;
      for (idx_t c = 0; c < IC; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            max_value = max_i(max_value, rgb[c][i][j]);
          }
        }
      }
      float l_max = 1.0 / (float)max_value;
      dataset[k].index = start + k;
      dataset[k].label = label[0];
      for (idx_t c = 0; c < IC; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            dataset[k](c,i,j) = rgb[c][i][j] * l_max;
          }
        }
      }
    }
    fclose(fp);
    /* choose holdout data */
    rnd_gen_t rgv;
    rgv.seed(validate_seed);
    for (int i = n_validate; i < n_data; i++) {
      if (rgv.rand(0.0, 1.0) * (i + 1) <= n_validate) {
        long v = rgv.randi(0, n_validate);
        cifar10_data_item<IC,H,W> d = dataset[v];
        dataset[v] = dataset[i];
        dataset[i] = d;
      }
    }    
    lgr.log(1, "loading data from %s [%d:%d] ends",
            cifar_bin, start, end);
    return 1;
  }
  int get_data_train(array4<maxB,IC,H,W>& x, ivec<maxB>& t, idx_t B) {
    assert(B <= maxB);
    x.set_n_rows(B);
    t.set_n(B);
    for (long b = 0; b < B; b++) {
      long idx = rg.randi(0, n_train);
      cifar10_data_item<IC,H,W>& itm = train[idx];
      t(b) = itm.label;
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            x(b,ic,i,j) = itm(ic,i,j);
          }
        }
      }
    }
    x.to_dev();
    t.to_dev();
    return 1;
  }
  int get_data_validate(array4<maxB,IC,H,W>& x, ivec<maxB>& t, idx_t from, idx_t to) {
    idx_t B = to - from;
    assert(B <= maxB);
    x.set_n_rows(B);
    t.set_n(B);
    for (long b = 0; b < B; b++) {
      cifar10_data_item<IC,H,W>& itm = validate[from + b];
      t(b) = itm.label;
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            x(b,ic,i,j) = itm(ic,i,j);
          }
        }
      }
    }
    x.to_dev();
    t.to_dev();
    return 1;
  }
};

int cifar_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = opt.batch_sz;
  const idx_t C = 64;
  const idx_t H = 32;
  const idx_t W = 32;
  logger lgr;
  lgr.start_log(opt);
  cifar10_dataset<maxB,C,H,W> ds;
  ds.set_seed(opt.sample_seed);
  ds.get_n_data_in_file(opt.cifar_data);
  ds.load(lgr, opt.cifar_data, 0, 0, 0, opt.validate_seed);
  array4<maxB,C,H,W> x;
  ivec<maxB> t;
  x.init_const(B, 0);
  t.init_const(B, 0);
  ds.get_data_train(x, t, opt.batch_sz);
  lgr.end_log();
  return 0;
}

