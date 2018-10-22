/**
   @file spmv.cc
   @brief sparse matrix vector multiplication
   @author Kenjiro Taura
   @date Oct. 14, 2018
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>

#if __NVCC__
/* cuda_util.h incudes various utilities to make CUDA 
   programming less error-prone. check it before you

   proceed with rewriting it for CUDA */
#include "cuda_util.h"
#endif

/** @brief type of matrix index (i,j,...)
    @details for large matrices, we might want to make it 64 bits
 */
typedef int idx_t;
/** @brief type of a matrix element */
typedef double real;

/** @brief sparse matrix storage format */
typedef enum {
  sparse_format_coo,        /**< coordinate list */
  sparse_format_coo_sorted, /**< sorted coordinate list */
  sparse_format_csr,        /**< compressed sparse row */
  sparse_format_invalid,    /**< invalid */
} sparse_format_t;

/** @brief type of sparse matrix we work on */
typedef enum {
  sparse_matrix_type_random, /**< uniform random matrix */
  sparse_matrix_type_rmat,   /**< R-MAT (recursive random matrix) */
  sparse_matrix_type_all_one, /**< all one (recursive random matrix) */
  sparse_matrix_type_coo_file, /**< input from file */
  sparse_matrix_type_invalid, /**< invalid */
} sparse_matrix_type_t;

/** @brief spmv matrix algorithm */
typedef enum {
  spmv_algo_serial,             /**< serial */
  spmv_algo_parallel,           /**< simple parallel */
  spmv_algo_cuda,               /**< cuda */
  spmv_algo_invalid             /**< invalid */
} spmv_algo_t;

/** @brief an element of coordinate list (i, j, a) */
typedef struct {
  idx_t i;                      /**< row */
  idx_t j;                      /**< column */
  real a;                       /**< element */
} coo_elem_t;

/** @brief an element of compressed sparse row */
typedef struct {
  idx_t j;                      /**< column */
  real a;                       /**< element */
} csr_elem_t;

/** @brief sparse matrix in coodinate list format */
typedef struct {
  coo_elem_t * elems;           /**< elements array */
#ifdef __NVCC__                 /* defined when compiling with nvcc */
  coo_elem_t * elems_dev;       /**< copy of elems on device */
#endif
} coo_t;

/** @brief sparse matrix in compressed row format */
typedef struct {
  idx_t * row_start;            /**< elems[row_start[i]] is the first element of row i */
  csr_elem_t * elems;           /**< elements array */
#ifdef __NVCC__
  idx_t * row_start_dev;        /**< copy of row_start on device */
  csr_elem_t * elems_dev;       /**< copy of elems on device */
#endif
} csr_t;

/** @brief sparse matrix (in any format) */
typedef struct {
  sparse_format_t format;  /**< format */
  idx_t M;                 /**< number of rows */
  idx_t N;                 /**< number of columns */
  idx_t nnz;               /**< number of non-zeros */
  union {
    coo_t coo;             /**< coo or sorted coo */
    csr_t csr;             /**< csr */
  };
} sparse_t;

/** @brief vector */
typedef struct {
  idx_t n;                 /**< number of elements */
  real * elems;            /**< array of elements */
#ifdef __NVCC__
  real * elems_dev;        /**< copy of elems on device */
#endif
} vec_t;

/** 
    @brief command line option
*/
typedef struct {
  idx_t M;                 /**< number of rows */
  idx_t N;                 /**< number of columns */
  idx_t nnz;               /**< number of non-zero elements */
  long repeat;             /**< number of iterations (tA (Ax)) */
  char * format_str;       /**< format string (coo, coo_sorted, csr) */
  sparse_format_t format;  /**< format_str converted to enum */

  char * matrix_type_str;  /**< matrix type (random, rmat, file) */
  sparse_matrix_type_t matrix_type; /**< matrix_type_str converted to enum */

  char * algo_str;         /**< algorithm string (serial, parallel, cuda) */
  spmv_algo_t algo;        /**< algo_str converted to enum */

  char * coo_file;         /**< file */
  char * rmat_str;         /**< a,b,c,d probability of rmat */
  double rmat[2][2];       /**< { { a, b }, { c, d } } probability of rmat */
  char * dump;             /**< file name to dump image (gnuplot) data */
  idx_t img_M;             /**< number of rows in the dumped image */
  idx_t img_N;             /**< number of columns in the dumped image */
  long seed;               /**< random number generator seed */
  int error;               /**< set when we encounter an error */
  int help;                /**< set when -h / --help is given */
} cmdline_options_t;

/** 
    @brief default values for command line options
*/
cmdline_options_t default_opts() {
  cmdline_options_t opt = {
    .M = 100000,
    .N = 0,
    .nnz = 0,
    .repeat = 5,
    .format_str = strdup("coo"),
    .format = sparse_format_invalid,
    .matrix_type_str = strdup("random"),
    .matrix_type = sparse_matrix_type_invalid,
    .algo_str = strdup("serial"),
    .algo = spmv_algo_invalid,
    .coo_file = strdup("mat.txt"),
    .rmat_str = strdup("4,1,2,3"),
    .rmat = { { 0, 0, }, { 0, 0, } },
    .dump = 0,
    .img_M = 512,
    .img_N = 512,
    .seed = 4567890123,
    .error = 0,
    .help = 0,
  };
  return opt;
}

/** 
    @brief command line options
*/
struct option long_options[] = {
  {"M",           required_argument, 0,  'M' },
  {"N",           required_argument, 0,  'N' },
  {"nnz",         required_argument, 0,  'z' },
  {"repeat",      required_argument, 0,  'r' },
  {"format",      required_argument, 0,  'f' },
  {"matrix-type", required_argument, 0,  't' },
  {"algo",        required_argument, 0,  'a' },
  {"coo-file",    required_argument, 0,   0  },
  {"rmat",        required_argument, 0,   0  },
  {"dump",        required_argument, 0,   0  },
  {"img-M",       required_argument, 0,   0  },
  {"img-N",       required_argument, 0,   0  },
  {"seed",        required_argument, 0,  's'},
  {"help",        required_argument, 0,  'h'},
  {0,             0,                 0,  0 }
};

/**
   @brief usage
  */
void usage(const char * prog) {
  cmdline_options_t o = default_opts();
  fprintf(stderr,
          "usage:\n"
          "\n"
          "%s [options ...]\n"
          "\n"
          "options:\n"
          "  --help             show this help\n"
          "  --M N              set the number of rows to N [%ld]\n"
          "  --N N              set the number of colums to N [%ld]\n"
          "  -z,--nnz N         set the number of non-zero elements to N [%ld]\n"
          "  -r,--repeat N      repeat N times [%ld]\n"
          "  -f,--format F      set sparse matrix format to F [%s]\n"
          "  -t,--matrix-type M set matrix type to T [%s]\n"
          "  -a,--algo A        set algorithm to A [%s]\n"
          "  --coo-file F       read matrix from F [%s]\n"
          "  --rmat a,b,c,d     set rmat probability [%s]\n"
          "  --dump F           dump matrix to image (gnuplot) file [%s]\n"
          "  --img-M M          number of rows in the dumped image [%ld]\n"
          "  --img-N N          number of columns in the dumped image [%ld]\n"
          "  -s,--seed S        set random seed to S [%ld]\n"
          ,
          prog,
          (long)o.M, (long)o.N,
          (long)o.nnz, o.repeat,
          o.format_str, o.matrix_type_str, o.algo_str,
          (o.coo_file ? o.coo_file : ""),
          o.rmat_str,
          (o.dump ? o.dump : ""),
          (long)o.img_M, (long)o.img_N, 
          o.seed);
}

/** 
    @brief parse a string for matrix format and return an enum value
*/
sparse_format_t parse_sparse_format(char * s) {
  if (strcasecmp(s, "coo") == 0) {
    return sparse_format_coo;
  } else if (strcasecmp(s, "coo_sorted") == 0) {
    return sparse_format_coo_sorted;
  } else if (strcasecmp(s, "csr") == 0) {
    return sparse_format_csr;
  } else {
    fprintf(stderr,
            "error:%s:%d: invalid sparse format (%s)\n",
            __FILE__, __LINE__, s);
    fprintf(stderr, "  must be one of { coo, coo_sorted, csr }\n");
    return sparse_format_invalid;
  }
}

/** 
    @brief parse a string for matrix format and return an enum value
*/
sparse_matrix_type_t parse_sparse_matrix_type(char * s) {
  if (strcasecmp(s, "random") == 0) {
    return sparse_matrix_type_random;
  } else if (strcasecmp(s, "rmat") == 0) {
    return sparse_matrix_type_rmat;
  } else if (strcasecmp(s, "one") == 0) {
    return sparse_matrix_type_all_one;
  } else if (strcasecmp(s, "coo_file") == 0) {
    return sparse_matrix_type_coo_file;
  } else {
    fprintf(stderr,
            "error:%s:%d: invalid matrix type (%s)\n",
            __FILE__, __LINE__, s);
    fprintf(stderr, "  must be one of { random, rmat, coo_file }\n");
    return sparse_matrix_type_invalid;
  }
}

/** 
    @brief parse a string for spmv algo and return an enum value
*/
spmv_algo_t parse_spmv_algo(char * s) {
  if (strcasecmp(s, "serial") == 0) {
    return spmv_algo_serial;
  } else if (strcasecmp(s, "parallel") == 0) {
    return spmv_algo_parallel;
  } else if (strcasecmp(s, "cuda") == 0) {
    return spmv_algo_cuda;
  } else {
    fprintf(stderr,
            "error:%s:%d: invalid spmv algo (%s)\n",
            __FILE__, __LINE__, s);
    fprintf(stderr, "  must be one of { serial, parallel }\n");
    return spmv_algo_invalid;
  }
}

/** 
    @brief print error meessage during rmat string (a,b,c,d)
*/
void parse_error_rmat_probability(char * rmat_str) {
  fprintf(stderr,
          "error:%s:%d: argument to --rmat (%s)"
          " must be F,F,F,F where F is a floating point number\n",
          __FILE__, __LINE__, rmat_str);
}
/** 
    @brief parse a string of the form a,b,c,d and put it into 2x2 matrix
*/
int parse_rmat_probability(char * rmat_str, double rmat[2][2]) {
  char * s = rmat_str;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i + j > 0) {
        if (s[0] != ',') {
          parse_error_rmat_probability(rmat_str);
          return 0;
        }
        s++;
      }
      char * next = 0;
      double x = strtod(s, &next);
      if (s == next) {
        /* no conversion performed */
        parse_error_rmat_probability(rmat_str);
        return 0;
      } else {
        rmat[i][j] = x;
        s = next;
      }
    }
  }
  double t = 0.0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      t += rmat[i][j];
    }
  }
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      rmat[i][j] /= t;
    }
  }
  return 1;
}

/** 
    @brief parse command line args
*/
void xfree(void*);
cmdline_options_t parse_args(int argc, char ** argv) {
  char * prog = argv[0];
  cmdline_options_t opt = default_opts();
  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "M:N:z:r:f:t:a:s:h",
                        long_options, &option_index);
    if (c == -1) break;
    switch (c) {
    case 0:
      {
        const char * o = long_options[option_index].name;
        if (strcmp(o, "rmat") == 0) {
          xfree(opt.rmat_str);
          opt.rmat_str = strdup(optarg);
        } else if (strcmp(o, "coo-file") == 0) {
          xfree(opt.coo_file);
          opt.coo_file = strdup(optarg);
        } else if (strcmp(o, "dump") == 0) {
          if (opt.dump) {
            xfree(opt.dump);
          }
          opt.dump = strdup(optarg);
        } else if (strcmp(o, "img-M") == 0) {
          opt.img_M = atol(optarg);
        } else if (strcmp(o, "img-N") == 0) {
          opt.img_N = atol(optarg);
        } else {
          fprintf(stderr,
                  "bug:%s:%d: should handle option %s\n",
                  __FILE__, __LINE__, o);
          opt.error = 1;
          return opt;
        }
      }
      break;
    case 'M':
      opt.M = atol(optarg);
      break;
    case 'N':
      opt.N = atol(optarg);
      break;
    case 'z':
      opt.nnz = atol(optarg);
      break;
    case 'r':
      opt.repeat = atol(optarg);
      break;
    case 'f':
      xfree(opt.format_str);
      opt.format_str = strdup(optarg);
      break;
    case 't':
      xfree(opt.matrix_type_str);
      opt.matrix_type_str = strdup(optarg);
      break;
    case 'a':
      xfree(opt.algo_str);
      opt.algo_str = strdup(optarg);
      break;
    case 's':
      opt.seed = atol(optarg);
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
  opt.format = parse_sparse_format(opt.format_str);
  if (opt.format == sparse_format_invalid) {
    opt.error = 1;
    return opt;
  }
  opt.matrix_type = parse_sparse_matrix_type(opt.matrix_type_str);
  if (opt.matrix_type == sparse_matrix_type_invalid) {
    opt.error = 1;
    return opt;
  }
  opt.algo = parse_spmv_algo(opt.algo_str);
  if (opt.algo == spmv_algo_invalid) {
    opt.error = 1;
    return opt;
  }
  if (parse_rmat_probability(opt.rmat_str, opt.rmat) == 0) {
    opt.error = 1;
    return opt;
  }
  return opt;
}



/**
   @brief malloc + check
   @param (sz) size to alloc in bytes
   @return pointer to the allocated memory
   @sa xfree
 */

void * xalloc(size_t sz) {
  void * a = malloc(sz);
  if (!a) {
    perror("malloc");
    exit(1);
  }
  return a;
}

/**
   @brief wrap free
   @param (a) a pointer returned by calling xalloc
   @sa xalloc
 */
void xfree(void * a) {
  free(a);
}

/** 
    @brief make an invalid matrix
*/
sparse_t mk_sparse_invalid() {
  sparse_t A = { sparse_format_invalid, 0, 0, 0, { } };
  return A;
}

/** 
    @brief destroy coo 
*/
void coo_destroy(sparse_t A) {
  xfree(A.coo.elems);
}

/** 
    @brief destroy csr
*/
void csr_destroy(sparse_t A) {
  xfree(A.csr.elems);
}

/** 
    @brief destroy sparse matrix in any format
*/
void sparse_destroy(sparse_t A) {
  switch (A.format) {
  case sparse_format_coo:
  case sparse_format_coo_sorted:
    coo_destroy(A);
    break;
  case sparse_format_csr:
    csr_destroy(A);
    break;
  default:
    fprintf(stderr,
            "error:%s:%d: sparse_destroy: invalid format %d\n",
            __FILE__, __LINE__, A.format);
    break;
  }
}

/**
   @brief destroy vector
 */
void vec_destroy(vec_t x) {
  xfree(x.elems);
}

/** 
    @brief make a uniform random coo matrix
*/
sparse_t mk_coo_random(idx_t M, idx_t N, idx_t nnz,
                       unsigned short rg[3]) {
  coo_elem_t * elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
  for (idx_t k = 0; k < nnz; k++) {
    idx_t i = nrand48(rg) % M;
    idx_t j = nrand48(rg) % N;
    real  a = erand48(rg);
    coo_elem_t * e = elems + k;
    e->i = i;
    e->j = j;
    e->a = a;
  }
  coo_t coo = { elems };
  sparse_t A = { sparse_format_coo, M, N, nnz, { .coo = coo } };
  return A;
}

typedef struct {
  idx_t i, j;
} idx_pair_t;

/**
   @brief return a pair of 0/1 ({0,0}, {0,1}, {1,0}, {1,1})
   according to 2x2 probability matrix p[2][2]
 */
idx_pair_t rmat_choose_01(double p[2][2], unsigned short rg[3]) {
  double x = erand48(rg);
  double q = 0.0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      q += p[i][j];
      if (x <= q) {
        idx_pair_t ij = { i, j };
        return ij;
      }
    }
  }
  idx_pair_t ij = { 1, 1 };
  return ij;
}

/** 
    @brief choose (i,j) s.t. 0 <= i < M and 0 <= j < N
    for R-MAT.
    @details
    according to the probability p[2][2];
    p[0][0] is the probability that it is chosen from the
    upper left (i.e., 0 <= i < M/2 and 0 <= j < N/2),
    p[0][1] the probability that it is chosen from the
    upper right (i.e., 0 <= i < M/2 and 0 <= j < N/2), and so on
*/
idx_pair_t rmat_choose_pair(idx_t M, idx_t N, double p[2][2],
                            unsigned short rg[3]) {
  idx_t M0 = 0, M1 = M, N0 = 0, N1 = N;
  while (M1 - M0 > 1 || N1 - N0 > 1) {
    idx_pair_t zo = rmat_choose_01(p, rg);
    if (M1 - M0 > 1) {
      idx_t Mh = (M0 + M1) / 2;
      if (zo.i) {
        M0 = Mh;
      } else {
        M1 = Mh;
      }
    }
    if (N1 - N0 > 1) {
      idx_t Nh = (N0 + N1) / 2;
      if (zo.j) {
        N0 = Nh;
      } else {
        N1 = Nh;
      }
    }
  }
  assert(M0 + 1 == M1);
  assert(N0 + 1 == N1);
  idx_pair_t ij = { M0, N0 };
  return ij;
}

/** 
    @brief make a random R-MAT 
    (https://epubs.siam.org/doi/abs/10.1137/1.9781611972740.43)
*/
sparse_t mk_coo_rmat(idx_t M, idx_t N, idx_t nnz,
                     double p[2][2], 
                     unsigned short rg[3]) {
  coo_elem_t * elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
  for (idx_t k = 0; k < nnz; k++) {
    idx_pair_t ij = rmat_choose_pair(M, N, p, rg);
    coo_elem_t * e = elems + k;
    e->i = ij.i;
    e->j = ij.j;
    e->a = erand48(rg);
  }
  coo_t coo = { elems };
  sparse_t A = { sparse_format_coo, M, N, nnz, { .coo = coo } };
  return A;
}

/** 
    @brief make a uniform random coo matrix
*/
sparse_t mk_coo_all_one(idx_t M, idx_t N, idx_t nnz) {
  idx_t nnz_M = 0;
  idx_t nnz_N = 0;
  int cont = 1;
  while (cont) {
    cont = 0;
    if (nnz_M < M && (nnz_M + 1) * nnz_N <= nnz) {
      nnz_M++;
      cont = 1;
    }
    if (nnz_N < N && nnz_M * (nnz_N + 1) <= nnz) {
      nnz_N++;
      cont = 1;
    }
  }
  idx_t real_nnz = nnz_M * nnz_N;
  assert(real_nnz <= nnz);
  assert(nnz_M <= M);
  assert(nnz_N <= N);
  if (real_nnz < nnz) {
    fprintf(stderr,
            "warning:%s:%d: nnz truncated to %ld\n",
            __FILE__, __LINE__, (long)real_nnz);
  }
  coo_elem_t * elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * real_nnz);
  idx_t skip_M = M / nnz_M;
  idx_t skip_N = N / nnz_N;
  idx_t k = 0;
  for (idx_t i = 0; i < nnz_M; i++) {
    for (idx_t j = 0; j < nnz_N; j++) {
      real  a = 1.0;
      coo_elem_t * e = elems + k;
      e->i = i * skip_M;
      e->j = j * skip_N;
      e->a = a;
      k++;
    }
  }
  assert(k == real_nnz);
  coo_t coo = { elems };
  sparse_t A = { sparse_format_coo_sorted, M, N, real_nnz, { .coo = coo } };
  return A;
}

/** 
    @brief compare two coo elements 
*/
int coo_elem_cmp(const void * a_, const void * b_) {
  coo_elem_t * a = (coo_elem_t *)a_;
  coo_elem_t * b = (coo_elem_t *)b_;
  if (a->i < b->i) return -1;
  if (a->i > b->i) return 1;
  if (a->j < b->j) return -1;
  if (a->j > b->j) return 1;
  return 0;
}

/** 
    @brief
    convert A to coo_sorted format.
    if in_place is true, update elements of A in place.
 */
sparse_t sparse_coo_to_coo_sorted(sparse_t A, int in_place) {
  if (A.format == sparse_format_coo
      || A.format == sparse_format_coo_sorted) {
    idx_t nnz = A.nnz;
    coo_elem_t * B_elems = 0;
    if (in_place) {
      B_elems = A.coo.elems;
    } else {
      B_elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
      memcpy(B_elems, A.coo.elems, sizeof(coo_elem_t) * nnz);
    }
    if (A.format == sparse_format_coo) {
      qsort((void*)B_elems, nnz, sizeof(coo_elem_t), coo_elem_cmp);
    }
    coo_t coo = { B_elems };
    sparse_t B = { sparse_format_coo_sorted, A.M, A.N, A.nnz, { .coo = coo } };
    return B;
  } else {
    fprintf(stderr,
            "error:%s:%d: input matrix not in coo format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
}

/**
   @brief coo -> csr
   @details if update_A is true, A's elements will become sorted
   in place as a side effect
 */
sparse_t sparse_coo_to_csr(sparse_t A, int update_A) {
  if (A.format == sparse_format_coo ||
      A.format == sparse_format_coo_sorted) {
    sparse_t B = sparse_coo_to_coo_sorted(A, update_A);
    idx_t M = B.M;
    idx_t N = B.N;
    idx_t nnz = B.nnz;
    idx_t * row_start = (idx_t *)xalloc(sizeof(idx_t) * (M + 1));
    coo_elem_t * B_elems = B.coo.elems;
    csr_elem_t * C_elems = (csr_elem_t *)xalloc(sizeof(csr_elem_t) * nnz);
    for (idx_t i = 0; i < M + 1; i++) {
      row_start[i] = 0;
    }
    for (idx_t k = 0; k < nnz; k++) {
      coo_elem_t * e = B_elems + k;
      row_start[e->i]++;
      C_elems[k].j = e->j;
      C_elems[k].a = e->a;
    }
    idx_t s = 0;
    for (idx_t i = 0; i < M; i++) {
      idx_t t = s + row_start[i];
      row_start[i] = s;
      s = t;
    }
    row_start[M] = s;
    assert(s == nnz);
    csr_t csr = { row_start, C_elems };
    sparse_t C = { sparse_format_csr, M, N, nnz, { .csr = csr } };
    return C;
  } else {
    fprintf(stderr,
            "error:%s:%d: input matrix not in coo format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
}

/**
   @brief convert sparse matrix in coo format to any specified format
 */
sparse_t sparse_coo_to(sparse_t A, sparse_format_t format, int update_A) {
  if (A.format == sparse_format_coo
      || A.format == sparse_format_coo_sorted) {
    switch (format) {
    case sparse_format_coo:
      return A;
    case sparse_format_coo_sorted:
      return sparse_coo_to_coo_sorted(A, update_A);
    case sparse_format_csr:
      return sparse_coo_to_csr(A, update_A);
    default:
      fprintf(stderr,
              "error:%s:%d: invalid output format %d\n",
              __FILE__, __LINE__, format);
      return mk_sparse_invalid();
    }
  } else {
    fprintf(stderr,
            "error:%s:%d: input matrix not in coo format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
}


/**
   @brief csr -> coo
 */
sparse_t sparse_csr_to_coo_sorted(sparse_t A) {
  if (A.format == sparse_format_csr) {
    idx_t M = A.M;
    idx_t N = A.N;
    idx_t nnz = A.nnz;
    idx_t * row_start = A.csr.row_start;
    csr_elem_t * A_elems = A.csr.elems;
    coo_elem_t * B_elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
    for (idx_t i = 0; i < M; i++) {
      idx_t start = row_start[i];
      idx_t end = row_start[i + 1];
      for (idx_t k = start; k < end; k++) {
        csr_elem_t * e = A_elems + k;
        B_elems[k].i = i;
        B_elems[k].j = e->j;
        B_elems[k].a = e->a;
      }
    }
    coo_t coo = { B_elems };
    sparse_t B = { sparse_format_coo_sorted, M, N, nnz, { .coo = coo } };
    return B;
  } else {
    fprintf(stderr,
            "error:%s:%d: input matrix not in csr format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
}


/**
   @brief convert sparse matrix in coo format to any specified format
 */
sparse_t sparse_csr_to(sparse_t A, sparse_format_t format) {
  if (A.format == sparse_format_csr) {
    switch (format) {
    case sparse_format_coo:
      return sparse_csr_to_coo_sorted(A);
    case sparse_format_coo_sorted:
      return sparse_csr_to_coo_sorted(A);
    case sparse_format_csr:
      return A;
    default:
      fprintf(stderr,
              "error:%s:%d: invalid output format %d\n",
              __FILE__, __LINE__, format);
      return mk_sparse_invalid();
    }
  } else {
    fprintf(stderr,
            "error:%s:%d: input matrix not in csr format %d\n",
            __FILE__, __LINE__, format);
    return mk_sparse_invalid();
  }
}

/**
   @brief convert any sparse matrix to coo format
 */
sparse_t sparse_to_coo(sparse_t A) {
  switch (A.format) {
  case sparse_format_coo:
    return A;
  case sparse_format_coo_sorted:
    return A;
  case sparse_format_csr:
    return sparse_csr_to_coo_sorted(A);
  default:
    fprintf(stderr,
            "error:%s:%d: invalid input format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
}
  
/**
   @brief convert any sparse matrix to coo sorted format
 */
sparse_t sparse_to_coo_sorted(sparse_t A, int update_A) {
  switch (A.format) {
  case sparse_format_coo:
    return sparse_coo_to_coo_sorted(A, update_A);
  case sparse_format_coo_sorted:
    return A;
  case sparse_format_csr:
    return sparse_coo_to_csr(A, update_A);
  default:
    fprintf(stderr,
            "error:%s:%d: invalid input format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
}
  
/**
   @brief convert any sparse matrix to csr format
 */
sparse_t sparse_to_csr(sparse_t A, int update_A) {
  switch (A.format) {
  case sparse_format_coo:
    return sparse_coo_to_csr(A, update_A);
  case sparse_format_coo_sorted:
    return sparse_coo_to_csr(A, update_A);
  case sparse_format_csr:
    return A;
  default:
    fprintf(stderr,
            "error:%s:%d: invalid input format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
}
  
/**
   @brief read a coo file
 */
sparse_t read_coo_file(idx_t M, idx_t N, idx_t nnz, char * file) {
  (void)M;
  (void)N;
  (void)nnz;
  (void)file;
  fprintf(stderr, "error:%s:%d: sorry, not implemented\n",
          __FILE__, __LINE__);
  return mk_sparse_invalid();
}

/**
   @brief make a sparse matrix in coo format
 */
sparse_t mk_sparse_matrix_coo(cmdline_options_t opt,
                              idx_t M, idx_t N, idx_t nnz,
                              unsigned short rg[3]) {
  switch (opt.matrix_type) {
  case sparse_matrix_type_random:
    return mk_coo_random(M, N, nnz, rg);
  case sparse_matrix_type_rmat:
    return mk_coo_rmat(M, N, nnz, opt.rmat, rg);
  case sparse_matrix_type_all_one:
    return mk_coo_all_one(M, N, nnz);
  case sparse_matrix_type_coo_file:
    return read_coo_file(M, N, nnz, opt.coo_file);
  default:
    fprintf(stderr,
            "error:%s:%d: invalid matrix_type %d\n",
            __FILE__, __LINE__, opt.matrix_type);
    return mk_sparse_invalid();
  }
}

/** 
    @brief make (read or generate) a sparse matrix with the specified method

    @param (opt) command line options
    @param (M) number of rows
    @param (N) number of columns
    @param (nnz) number of non-zeros
    @param (rg) random number generator state

    @details make a MxN sparse matrix with nnz non-zero elements
    either by generating one or reading one from a file.
*/
sparse_t mk_sparse_matrix(cmdline_options_t opt,
                          idx_t M, idx_t N, idx_t nnz,
                          unsigned short rg[3]) {
  sparse_t A = mk_sparse_matrix_coo(opt, M, N, nnz, rg);
  sparse_t B = sparse_coo_to(A, opt.format, 1);
  return B;
}

/** 
    @brief transpose a matrix in coordinate list format
*/
sparse_t coo_transpose(sparse_t A, int in_place) {
  assert(A.format == sparse_format_coo
         || A.format == sparse_format_coo_sorted);
  idx_t nnz = A.nnz;
  coo_elem_t * B_elems = 0;
  if (in_place) {
    B_elems = A.coo.elems;
  } else {
    B_elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
    memcpy(B_elems, A.coo.elems, sizeof(coo_elem_t) * nnz);
  }
  for (idx_t k = 0; k < nnz; k++) {
    idx_t i = B_elems[k].i;
    idx_t j = B_elems[k].j;
    B_elems[k].i = j;
    B_elems[k].j = i;
  }
  coo_t coo = { B_elems };
  sparse_t B = { sparse_format_coo, A.N, A.M, nnz, { .coo = coo } };
  return B;
}

/** 
    @brief transpose a matrix in any format
*/
sparse_t sparse_transpose(sparse_t A) {
  switch (A.format) {
  case sparse_format_coo: {
    return coo_transpose(A, 0);
  }
  case sparse_format_coo_sorted: {
    sparse_t B = coo_transpose(A, 0);
    return sparse_coo_to_coo_sorted(B, 1);
  }
  case sparse_format_csr: {
    sparse_t B = sparse_csr_to_coo_sorted(A);
    sparse_t C = coo_transpose(B, 1);
    sparse_t D = sparse_coo_to_csr(C, 1);
    sparse_destroy(B);          // and C
    return D;
  }
  default: {
    fprintf(stderr,
            "error:%s:%d: invalid format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
  }
}

#if __NVCC__
int coo_to_dev(sparse_t& A) {
  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d: write a code that copies the coo elements of A to the device.\n"
          "use dev_malloc and to_dev utility functions in cuda_util.h\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);
  return 1;
}

int csr_to_dev(sparse_t& A) {
  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d: write a code that copies the csr elements of A to the device.\n"
          "use dev_malloc and to_dev utility functions in cuda_util.h\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);
  return 1;
}


/** 
    @brief copy elements to device
*/
int sparse_to_dev(sparse_t& A) {
  switch (A.format) {
  case sparse_format_coo: {
    return coo_to_dev(A);
  }
  case sparse_format_coo_sorted: {
    return coo_to_dev(A);
  }
  case sparse_format_csr: {
    return csr_to_dev(A);
  }
  default: {
    fprintf(stderr,
            "error:%s:%d: invalid format %d\n",
            __FILE__, __LINE__, A.format);
    return 0;
  }
  }
}

int vec_to_dev(vec_t& v) {
  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d: write a code that copies the elements of v to the device.\n"
          "use dev_malloc and to_dev utility functions in cuda_util.h\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  return 1;
}
#endif


/** 
    @brief y = A * x in serial for coordinate list format
*/
long spmv_coo_serial(sparse_t A, vec_t vx, vec_t vy) {
  idx_t M = A.M;
  idx_t nnz = A.nnz;
  coo_elem_t * elems = A.coo.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
  for (idx_t k = 0; k < nnz; k++) {
    coo_elem_t * e = elems + k;
    idx_t i = e->i;
    idx_t j = e->j;
    real  a = e->a;
    real ax = a * x[j];
    y[i] += ax;
  }
  return 2 * (long)nnz;
}

/** 
    @brief y = A * x in parallel for coordinate list format
*/
long spmv_coo_parallel(sparse_t A, vec_t vx, vec_t vy) {

  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d: write a code that performs SPMV with COO format in parallel\n"
          "using OpenMP parallel for directives.\n"
          "you need to insert a few programs into the serial version below.\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);
  
  idx_t M = A.M;
  idx_t nnz = A.nnz;
  coo_elem_t * elems = A.coo.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
  for (idx_t k = 0; k < nnz; k++) {
    coo_elem_t * e = elems + k;
    idx_t i = e->i;
    idx_t j = e->j;
    real  a = e->a;
    real ax = a * x[j];
    y[i] += ax;
  }
  return 2 * (long)nnz;
}

#if __NVCC__

// placeholders to define kernels for your convenience

__global__ void init_const_dev(vec_t v, real c) {

}

__global__ void spmv_coo_dev(sparse_t A, vec_t vx, vec_t vy) {

}

/** 
    @brief y = A * x with cuda for coordinate list format
*/
long spmv_coo_cuda(sparse_t A, vec_t vx, vec_t vy) {
  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d: write a code that performs SPMV with COO format in parallel\n"
          "using CUDA kernel functions.\n"
          "you need to convert the for loops to kernels + kernel launches.\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);

  idx_t M = A.M;
  idx_t nnz = A.nnz;
  coo_elem_t * elems = A.coo.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) { // convert this to kernel + kernel launch
    y[i] = 0.0;
  }
  for (idx_t k = 0; k < nnz; k++) { // convert this to kernel + kernel launch
    coo_elem_t * e = elems + k;
    idx_t i = e->i;
    idx_t j = e->j;
    real  a = e->a;
    real ax = a * x[j];
    y[i] += ax;
  }
  
  return 2 * (long)nnz;
}
#endif                                 

/** 
    @brief y = A * x for coordinate list format
*/
long spmv_coo(spmv_algo_t algo, sparse_t A, vec_t x, vec_t y) {
  switch (algo) {
  case spmv_algo_serial:
    return spmv_coo_serial(A, x, y);
  case spmv_algo_parallel:
    return spmv_coo_parallel(A, x, y);
#if __NVCC__
  case spmv_algo_cuda:
    return spmv_coo_cuda(A, x, y);
#endif
  default:
    fprintf(stderr,
            "error:%s:%d: invalid algorithm %d\n",
            __FILE__, __LINE__, algo);
    return -1;
  }
}

/** 
    @brief y = A * x in serial for csr format
*/
long spmv_csr_serial(sparse_t A, vec_t vx, vec_t vy) {
  idx_t M = A.M;
  idx_t nnz = A.nnz;
  idx_t * row_start = A.csr.row_start;
  csr_elem_t * elems = A.csr.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
  for (idx_t i = 0; i < M; i++) {
    idx_t start = row_start[i];
    idx_t end = row_start[i + 1];
    for (idx_t k = start; k < end; k++) {
      csr_elem_t * e = elems + k;
      idx_t j = e->j;
      real  a = e->a;
      y[i] += a * x[j];
    }
  }
  return 2 * (long)nnz;
}

/** 
    @brief y = A * x in parallel for csr format
*/
long spmv_csr_parallel(sparse_t A, vec_t vx, vec_t vy) {

  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d: write a code that performs SPMV with CSR format in parallel\n"
          "using CUDA kernel functions.\n"
          "you need to convert the for loops to kernels + kernel launches.\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);
  
  idx_t M = A.M;
  idx_t nnz = A.nnz;
  idx_t * row_start = A.csr.row_start;
  csr_elem_t * elems = A.csr.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
  for (idx_t i = 0; i < M; i++) {
    idx_t start = row_start[i];
    idx_t end = row_start[i + 1];
    for (idx_t k = start; k < end; k++) {
      csr_elem_t * e = elems + k;
      idx_t j = e->j;
      real  a = e->a;
      y[i] += a * x[j];
    }
  }

  return 2 * (long)nnz;
}

/** 
    @brief y = A * x with cuda for csr format
*/

// add necessary kernel definitions by yourself

long spmv_csr_cuda(sparse_t A, vec_t vx, vec_t vy) {

  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d: write a code that performs SPMV with CSR format in parallel\n"
          "using CUDA kernel functions.\n"
          "you need to convert the for loops to kernels + kernel launches.\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);
  
  idx_t M = A.M;
  idx_t nnz = A.nnz;
  idx_t * row_start = A.csr.row_start;
  csr_elem_t * elems = A.csr.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) { // conver to kernel + kernel launch
    y[i] = 0.0;
  }
  for (idx_t i = 0; i < M; i++) { // conver to kernel + kernel launch
    idx_t start = row_start[i];
    idx_t end = row_start[i + 1];
    for (idx_t k = start; k < end; k++) {
      csr_elem_t * e = elems + k;
      idx_t j = e->j;
      real  a = e->a;
      y[i] += a * x[j];
    }
  }

  return 2 * (long)nnz;
}



/** 
    @brief y = A * x for csr format
*/
long spmv_csr(spmv_algo_t algo, sparse_t A, vec_t x, vec_t y) {
  switch (algo) {
  case spmv_algo_serial:
    return spmv_csr_serial(A, x, y);
  case spmv_algo_parallel:
    return spmv_csr_parallel(A, x, y);
#if __NVCC__
  case spmv_algo_cuda:
    return spmv_csr_cuda(A, x, y);
#endif
  default:
    fprintf(stderr,
            "error:%s:%d: invalid algorithm %d\n",
            __FILE__, __LINE__, algo);
    return -1;
  }
}

/** 
    @brief y = A * x
*/
long spmv(spmv_algo_t algo, sparse_t A, vec_t x, vec_t y) {
  assert(x.n == A.N);
  assert(y.n == A.M);
  switch (A.format) {
  case sparse_format_coo:
    return spmv_coo(algo, A, x, y);
  case sparse_format_coo_sorted:
    return spmv_coo(algo, A, x, y);
  case sparse_format_csr:
    return spmv_csr(algo, A, x, y);
  default:
    fprintf(stderr,
            "error:%s:%d: invalid format %d\n",
            __FILE__, __LINE__, A.format);
    return -1;
  }
}

/** 
    @brief square norm of a vector (serial)
*/
real vec_norm2_serial(vec_t v) {
  real s = 0.0;
  real * x = v.elems;
  idx_t n = v.n;
  for (idx_t i = 0; i < n; i++) {
    s += x[i] * x[i];
  }
  return s;
}

/** 
    @brief square norm of a vector (parallel)
*/
real vec_norm2_parallel(vec_t v) {
  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d: write a code that computes square norm of a vector v\n"
          "using omp parallel for.\n"
          "you need to insert a few pragmas into the serial version below\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);

  real s = 0.0;
  real * x = v.elems;
  idx_t n = v.n;
  for (idx_t i = 0; i < n; i++) {
    s += x[i] * x[i];
  }
  return s;
}

#if __NVCC__
__global__ void vec_norm2_dev(vec_t v, real * s) {
  
}

/** 
    @brief square norm of a vector (parallel)
*/
real vec_norm2_cuda(vec_t v) {
  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d: write a code that computes square norm of a vector v\n"
          "using CUDA.\n"
          "you need to convert a loop to kernel + kernel launch\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);

  real s = 0.0;
  real * x = v.elems;
  idx_t n = v.n;
  for (idx_t i = 0; i < n; i++) {
    s += x[i] * x[i];
  }
  return s;
}
#endif

real vec_norm2(spmv_algo_t algo, vec_t v) {
  switch(algo) {
  case spmv_algo_serial:
    return vec_norm2_serial(v);
  case spmv_algo_parallel:
    return vec_norm2_parallel(v);
#if __NVCC__
  case spmv_algo_cuda:
    return vec_norm2_cuda(v);
#endif
  default:
    fprintf(stderr,
            "error:%s:%d: invalid algo %d\n",
            __FILE__, __LINE__, algo);
    return -1.0;
  }
}
  
/** 
    @brief scalar x vector (serial)
*/
int scalar_vec_serial(real k, vec_t v) {
  idx_t n = v.n;
  real * x = v.elems;
  for (idx_t i = 0; i < n; i++) {
    x[i] *= k;
  }
  return 1;
}

/** 
    @brief scalar x vector (parallel)
*/
int scalar_vec_parallel(real k, vec_t v) {
  idx_t n = v.n;
  real * x = v.elems;
#pragma omp parallel for
  for (idx_t i = 0; i < n; i++) {
    x[i] *= k;
  }
  return 1;
}

/** 
    @brief scalar x vector (cuda)
*/
#if __NVCC__
__global__ void scalar_vec_dev(real k, vec_t v) {
  idx_t n = v.n;
  real * x = v.elems_dev;
  idx_t i = get_thread_id_x();
  if (i < n) {
    x[i] *= k;
  }
}

int scalar_vec_cuda(real k, vec_t v) {
  idx_t n = v.n;
  int scalar_vec_block_sz = 1024;
  int n_scalar_vec_blocks = (n + scalar_vec_block_sz - 1) / scalar_vec_block_sz;
  check_launch_error((scalar_vec_dev<<<n_scalar_vec_blocks,scalar_vec_block_sz>>>(k, v)));
  return 1;
}
#endif

/**
   @brief scalar x vector
 */

int scalar_vec(spmv_algo_t algo, real k, vec_t v) {
  switch(algo) {
  case spmv_algo_serial:
    return scalar_vec_serial(k, v);
  case spmv_algo_parallel:
    return scalar_vec_parallel(k, v);
#if __NVCC__
  case spmv_algo_cuda:
    return scalar_vec_cuda(k, v);
#endif
  default:
    fprintf(stderr,
            "error:%s:%d: invalid algo %d\n",
            __FILE__, __LINE__, algo);
    return 0;
  }
}
  
/** 
    @brief normalize a vector
*/
real vec_normalize(spmv_algo_t algo, vec_t v) {
  real s2 = vec_norm2(algo, v);
  if (s2 < 0.0) return -1.0;
  real s = sqrt(s2);
  if (!scalar_vec(algo, 1/s, v)) return -1.0;
  return s;
}

/** 
    @brief current time in nano second
*/
long cur_time_ns() {
  struct timespec ts[1];
  clock_gettime(CLOCK_REALTIME, ts);
  return ts->tv_sec * 1000000000L + ts->tv_nsec;
}

/** 
    @brief repeat y = A x; x = tA y; many times
*/
real repeat_spmv(spmv_algo_t algo,
                 sparse_t& A, sparse_t& tA,
                 vec_t& x, vec_t& y, idx_t repeat) {
#if __NVCC__
  if (algo == spmv_algo_cuda) {
    sparse_to_dev(A);
    sparse_to_dev(tA);
    vec_to_dev(x);
    vec_to_dev(y);
  }
#endif
  
  printf("repeat_spmv : warm up + error check\n");
  fflush(stdout);
  if (spmv(algo, A, x, y) < 0.0) {              // y = A x
    return -1.0;
  }
  if (vec_norm2(algo, y) < 0.0) {
    return -1.0;
  }
  if (spmv(algo, tA, y, x) < 0.0) {              // y = A x
    return -1.0;
  }
  if (vec_normalize(algo, x) < 0.0) {
    return -1.0;
  }
  printf("repeat_spmv : start\n");
  fflush(stdout);
  real lambda = 0.0;
  long flops = 0;
  long t0 = cur_time_ns();
  for (idx_t r = 0; r < repeat; r++) {
    flops += spmv(algo,  A, x, y);
    lambda = sqrt(vec_norm2(algo, y));
    flops += spmv(algo, tA, y, x);
    vec_normalize(algo, x);
    flops += 2 * y.n + 3 * y.n;
  }
  long t1 = cur_time_ns();
  long dt = t1 - t0;
  printf("%ld flops in %.9e sec (%.9e GFLOPS)\n",
         flops, dt*1.0e-9, flops/(double)dt);
  return lambda;
}
  
/** 
    @brief make a random vector
*/
vec_t mk_vec_random(idx_t n, unsigned short rg[3]) {
  real * x = (real *)xalloc(sizeof(real) * n);
  for (idx_t i = 0; i < n; i++) {
    x[i] = erand48(rg);
  }
  vec_t v = { n, x };
  return v;
}

/** 
    @brief make a unit-length random vector
*/
vec_t mk_vec_unit_random(idx_t n, unsigned short rg[3]) {
  vec_t x = mk_vec_random(n, rg);
  vec_normalize(spmv_algo_serial, x);
  return x;
}

/** 
    @brief make a zero vector
*/
vec_t mk_vec_zero(idx_t n) {
  real * x = (real *)xalloc(sizeof(real) * n);
  for (idx_t i = 0; i < n; i++) {
    x[i] = 0.0;
  }
  vec_t v = { n, x };
  return v;
}


/** 
    @brief release memory for cmdline_options
*/
void cmdline_options_destroy(cmdline_options_t opt) {
  xfree(opt.format_str);
  xfree(opt.matrix_type_str);
  xfree(opt.algo_str);
  if (opt.coo_file) {
    xfree(opt.coo_file);
  }
  xfree(opt.rmat_str);
  if (opt.dump) {
    xfree(opt.dump);
  }
}

/**
   @brief data structure to dump matrix into a gnuplot file
 */
typedef struct {
  double * a;
  idx_t M;
  idx_t N;
  idx_t img_M;
  idx_t img_N;
  double scale_i;
  double scale_j;
} image_t;

/**
   @brief make an img_M x img_N image for M x N matrix
 */
image_t mk_image(idx_t M, idx_t N, idx_t img_M, idx_t img_N) {
  if (img_M == 0) img_M = M;
  if (img_N == 0) img_N = N;
  double scale_i = img_M / (double)M;
  double scale_j = img_N / (double)N;
  double * a = (double *)xalloc(sizeof(double) * img_M * img_N);
  for (idx_t i = 0; i < img_M; i++) {
    for (idx_t j = 0; j < img_N; j++) {
      a[i * img_N + j] = 0.0;
    }
  }
  image_t img = { a, M, N, img_M, img_N, scale_i, scale_j };
  return img;
}

void image_destroy(image_t * img) {
  xfree(img->a);
}

/**
   @brief add a pixel to an img.
 */
void image_add_pixel(image_t * img, idx_t i, idx_t j, double p) {
  idx_t min_img_i = (idx_t)(i * img->scale_i);
  idx_t min_img_j = (idx_t)(j * img->scale_j);
  idx_t max_img_i = (idx_t)((i + 1) * img->scale_i);
  idx_t max_img_j = (idx_t)((j + 1) * img->scale_j);
  if (min_img_i == max_img_i) max_img_i = min_img_i + 1;
  if (min_img_j == max_img_j) max_img_j = min_img_j + 1;
  idx_t img_M = img->img_M;
  idx_t img_N = img->img_N;
  assert(min_img_i < img_M);
  assert(min_img_j < img_N);
  assert(max_img_i <= img_M);
  assert(max_img_j <= img_N);
  for (idx_t x = min_img_i; x < max_img_i; x++) {
    for (idx_t y = min_img_j; y < max_img_j; y++) {
      if (0 <= x && x < img_M && 0 <= y && y < img_N) {
        (void)p;
        //img->a[x * img_N + y] += p;
        img->a[x * img_N + y] = 1;
      }
    }
  }
}

int dump_sparse_file(sparse_t A, idx_t img_M, idx_t img_N, char * file) {
  printf("dumping to %s (matrix size %ld x %ld -> image size: %ld x %ld)\n",
         file, (long)A.M, (long)A.N, (long)img_M, (long)img_N);
  sparse_t B = sparse_to_coo(A);
  idx_t nnz = B.nnz;
  coo_elem_t * elems = B.coo.elems;
  image_t img[1] = { mk_image(A.M, A.N, img_M, img_N) };
  for (idx_t k = 0; k < nnz; k++) {
    coo_elem_t * e = elems + k;
    image_add_pixel(img, e->i, e->j, e->a);
  }
  FILE * wp = fopen(file, "w");
  if (!wp) {
    perror(file);
    return 0;
  }
  fprintf(wp, "set title \"\"\n"); /* white -> blue */
  fprintf(wp, "set palette rgbformula -3,-3,2\n"); /* white -> blue */
  fprintf(wp, "$mat << EOD\n");
  idx_t k = 0;
  for (idx_t i = 0; i < img->img_M; i++) {
    for (idx_t j = 0; j < img->img_N; j++) {
      fprintf(wp, "%f ", img->a[k]);
      k++;
    }
    fprintf(wp, "\n");
  }
  fprintf(wp, "EOD\n");
  fprintf(wp, "plot '$mat' matrix using 1:2:3 with image\n");
  fclose(wp);
  image_destroy(img);
  return 1;
}

/** 
    @brief main
*/
int main(int argc, char ** argv) {
  cmdline_options_t opt = parse_args(argc, argv);
  if (opt.help || opt.error) {
    usage(argv[0]);
    exit(opt.error);
  }
  idx_t M        = opt.M;
  idx_t N        = (opt.N ? opt.N : M);
  idx_t nnz      = (opt.nnz ? opt.nnz : ((long)M * (long)N + 99L) / 100L);
  long repeat    = opt.repeat;
  unsigned short rg[3] = {
    (unsigned short)((opt.seed >> 32) & ((1 << 16) - 1)),
    (unsigned short)((opt.seed >> 16) & ((1 << 16) - 1)),
    (unsigned short)((opt.seed >> 0 ) & ((1 << 16) - 1)),
  };
  long flops = 2 * 2 * nnz * repeat;
  printf("A : %ld x %ld, %ld non-zeros %ld bytes for non-zeros\n",
         (long)M, (long)N, (long)nnz, nnz * sizeof(real));
  printf("repeat : %ld times\n", repeat);
  printf("format : %s\n", opt.format_str);
  printf("matrix : %s\n", opt.matrix_type_str);
  printf("algo : %s\n", opt.algo_str);
  printf("%ld flops for spmv\n", flops);

  //sparse_t A = mk_sparse_random(opt.format, M, N, nnz, rg);
  printf("generating %ld x %ld matrix (%ld non-zeros) ... ",
         (long)M, (long)N, (long)nnz);
  fflush(stdout);
  sparse_t A = mk_sparse_matrix(opt, M, N, nnz, rg);
  printf(" done\n"); fflush(stdout);
  if (opt.dump) {
    dump_sparse_file(A, opt.img_M, opt.img_N, opt.dump);
  }
  sparse_t tA = sparse_transpose(A);
  vec_t x = mk_vec_unit_random(N, rg);
  vec_t y = mk_vec_zero(M);
  real lambda = repeat_spmv(opt.algo, A, tA, x, y, repeat);
  if (lambda == -1.0) {
    printf("an error ocurred during repeat_spmv\n");
  } else {
    printf("lambda = %.9e\n", lambda);
  }
  vec_destroy(x);
  vec_destroy(y);
  sparse_destroy(A);
  sparse_destroy(tA);
  cmdline_options_destroy(opt);
  return 0;
}

