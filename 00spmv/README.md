
Build
=================

```
$ make
g++  -Wall -Wextra -O3 -fopenmp   -c -o spmv.o spmv.cc
g++ -o spmv spmv.o  -Wall -Wextra -O3 -fopenmp 
```

do this on the login node.

Run
=================

Example:

```
$ srun -p big ./spmv 
A : 100000 x 100000, 100000000 non-zeros 800000000 bytes for non-zeros
repeat : 5 times
1000000000 flops
repeat_spmv : warm up + error check
repeat_spmv : start
2002500000 flops in 6.514223060e+00 sec (3.074042724e-01 GFLOPS)
lambda = 5.006385423e+02
```

srun should be used to run any executable on a compute node.

But for a very small/short run for the purpose of quick correctness check, you can directly run it on the login node.

```
$ ./spmv 
A : 100000 x 100000, 100000000 non-zeros 800000000 bytes for non-zeros
repeat : 5 times
1000000000 flops
repeat_spmv : warm up + error check
repeat_spmv : start
2002500000 flops in 6.514223060e+00 sec (3.074042724e-01 GFLOPS)
lambda = 5.006385423e+02
```

How to modify the file
=================

