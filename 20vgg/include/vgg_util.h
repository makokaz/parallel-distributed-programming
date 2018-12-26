/**
   @file vgg_util.h
 */
#pragma once
#include <assert.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
   @brief type of array index (either int or long)
 */
typedef int idx_t;
/**
   @brief type of array elements
 */
typedef float real;

static void bail() {
  exit(1);
}

struct cmdline_opt {
  const char * cifar_data;      /**< data file */
  long start_data;              /**< read from this image */
  long end_data;                /**< read before this image */
  idx_t batch_sz;               /**< batch size */
  real learnrate;               /**< learning rate */
  long iters;                   /**< number of batches to process */
  long sample_seed;             /**< random seed to draw samples */
  long weight_seed;             /**< random seed to initialize weights and dropout */
  long dropout_seed;            /**< random seed to determine dropout */
  int verbose;                  /**< verbosity */
  int help;
  int error;
  cmdline_opt() {
    cifar_data = "cifar-10-batches-bin/data_batch_1.bin";
    start_data = 0;
    end_data = 0;
    batch_sz = MAX_BATCH_SIZE;
    learnrate = 0.05;
    iters = 2;
    sample_seed  = 34567890123452L;
    dropout_seed = 56789012345234L;
    weight_seed  = 45678901234523L;
    verbose = 2;
    help = 0;
    error = 0;
  }
};

static struct option long_options[] = {
  {"cifar_data",  required_argument, 0, 'd' },
  {"start_data",  required_argument, 0, 0 },
  {"end_data",    required_argument, 0, 0 },
  {"batch_sz",    required_argument, 0, 'b' },
  {"learnrate",   required_argument, 0, 'l' },
  {"iter",        required_argument, 0, 'm' },
  {"sample_seed", required_argument, 0, 's' },
  {"dropout_seed",required_argument, 0, 'S' },
  {"weight_seed", required_argument, 0, 'w' },
  {"verbose",     required_argument, 0, 'v'   },
  {"help",        required_argument, 0, 'h' },
  {0,             0,                 0,  0  }
};

static void usage(const char * prog) {
  fprintf(stderr,
          "usage:\n"
          "\n"
          "%s [options]\n"
          "\n"
          " -d,--cifar_data F : read data from F\n"
          " --start_data A : start reading from Ath image\n"
          " --end_data A : stop reading before Ath image\n"
          " -b,--batch_sz N : set batch size to N\n"
          " -l,--learnrate ETA : set learning rate to ETA\n"
          " -m,--iter N : iterate N times\n"
          " -s,--seed S : set seed to S\n"
          " -v,--verbose L : set verbosity level to L\n"
          " -h,--help\n",
          prog);
  exit(1);
}

static cmdline_opt parse_args(int argc, char ** argv) {
  char * prog = argv[0];
  cmdline_opt opt;
  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv,
                        "b:d:l:m:s:S:h", long_options, &option_index);
    if (c == -1) break;
    switch (c) {
    case 0:
      {
        const char * o = long_options[option_index].name;
        if (strcmp(o, "start_data") == 0) {
          opt.start_data = atol(optarg);
        } else if (strcmp(o, "end_data") == 0) {
          opt.end_data = atol(optarg);
        } else {
          fprintf(stderr,
                  "bug:%s:%d: should handle option %s\n",
                  __FILE__, __LINE__, o);
          opt.error = 1;
        }
        return opt;
      }
      break;
    case 'b':
      opt.batch_sz = atoi(optarg);
      break;
    case 'd':
      opt.cifar_data = strdup(optarg);
      break;
    case 'l':
      opt.learnrate = atof(optarg);
      break;
    case 'm':
      opt.iters = atol(optarg);
      break;
    case 's':
      opt.sample_seed = atol(optarg);
      break;
    case 'S':
      opt.dropout_seed = atol(optarg);
      break;
    case 'w':
      opt.weight_seed = atol(optarg);
      break;
    case 'v':
      opt.verbose = atoi(optarg);
      break;
    case 'h':
      opt.help = 1;
      break;
    default: /* '?' */
      usage(prog);
      opt.error = 1;
      return opt;
    }
  }
  return opt;
}

static real max_r(real a, real b) {
  return (a < b ? b : a);
}

static real min_r(real a, real b) {
  return (a < b ? a : b);
}

static idx_t max_i(idx_t a, idx_t b) {
  return (a < b ? b : a);
}

static idx_t min_i(idx_t a, idx_t b) {
  return (a < b ? a : b);
}

struct tsc_t {
  long ns;
};

static tsc_t get_tsc() {
  struct timespec ts[1];
  tsc_t t;
  if (clock_gettime(CLOCK_REALTIME, ts) == -1) {
    perror("clock_gettime"); bail();
  }
  t.ns = ts->tv_sec * 1000000000L + ts->tv_nsec;
  return t;
}

struct rnd_gen_t {
  unsigned short rg[3];
  void seed(long s) {
    rg[0] = (s >> 0)  & 65535;
    rg[1] = (s >> 16) & 65535;
    rg[2] = (s >> 32) & 65535;
  }
  double rand01() {
    return erand48(rg);
  }
  double rand(real p, real q) {
    return p + rand01() * (q - p);
  }
  long randi01() {
    return nrand48(rg);
  }
  long randi(long a, long b) {
    return a + randi01() % (b - a);
  }

  /* generate a random number from a normal distribution whose 
     mean is mu and variance is sigma^2. used to initialize
     the weight matrices. see
     https://en.wikipedia.org/wiki/Normal_distribution
     for how the following method works */
  real rand_normal(real mu, real sigma) {
    real u = erand48(rg);
    real v = erand48(rg);
    real x = sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
    return mu + x * sigma;
  }
  long get_state() {
    return ((long)rg[2] << 32) + ((long)rg[1] << 16) + (long)rg[0];
  }
};

int vgg_util_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  rnd_gen_t rg;
  rg.seed(opt.sample_seed);
  bail();
  min_i(1,2);
  max_i(3,4);
  min_r(1.2,3.4);
  max_r(5.6,7.8);
  return 0;
}

void vgg_util_use_unused_functions() {
  (void)get_tsc;
}
