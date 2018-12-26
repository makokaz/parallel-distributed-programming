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
  cifar10_data_item<IC,H,W> * dataset;
  long n_data;
  rnd_gen_t rg;
  void set_seed(long sd) {
    rg.seed(sd);
  }
  long get_n_data(const char * cifar_bin) {
    struct stat sb[1];
    if (lstat(cifar_bin, sb) == -1) {
      perror("lstat"); exit(1);
    }
    long sz = sb->st_size;
    long sz1 = IC * H * W + 1;
    assert(sz % sz1 == 0);
    return sz / sz1;
  }
  int load(const char * cifar_bin, long start, long end) {
    long n_data_in_file = get_n_data(cifar_bin);
    if (end == 0) end = n_data_in_file;
    assert(start >= 0);
    assert(end <= n_data_in_file);
    assert(start < end);

    n_data = end - start;
    dataset = new cifar10_data_item<IC,H,W> [n_data];
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
    return 1;
  }
  int get_data_train(array4<maxB,IC,H,W>& x, ivec<maxB>& t, idx_t B) {
    assert(B <= maxB);
    for (long b = 0; b < B; b++) {
      long idx = rg.randi01() % n_data;
      cifar10_data_item<IC,H,W>& itm = dataset[idx];
      t(b) = itm.label;
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            x(b,ic,i,j) = itm(ic,i,j);
          }
        }
      }
    }
    return 1;
  }
  int get_data_validate(array4<maxB,IC,H,W>& x, ivec<maxB>& t, idx_t B) {
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
  cifar10_dataset<maxB,C,H,W> ds;
  ds.set_seed(opt.sample_seed);
  ds.get_n_data(opt.cifar_data);
  ds.load(opt.cifar_data, 0, ds.get_n_data(opt.cifar_data));
  array4<maxB,C,H,W> x;
  ivec<maxB> t;
  x.init_const(B, 0);
  t.init_const(B, 0);
  ds.get_data_train(x, t, opt.batch_sz);
  return 0;
}

