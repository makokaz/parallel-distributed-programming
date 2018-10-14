/**
   @file hello.c
  */
#include <stdio.h>
#include <omp.h>

void worker() {
  int rank = omp_get_thread_num();
  int nthreads = omp_get_num_threads();
  printf("hello from thread %d of %d\n", rank, nthreads);
}

int main() {
  printf("hello before omp parallel\n");
#pragma omp parallel
  worker();
  printf("hello after omp parallel\n");
  return 0;
}
