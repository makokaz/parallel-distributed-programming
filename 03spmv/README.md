
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

help
=================

Just run

```
$ ./spmv -h
usage:

./spmv [options ...]

options:
  --help        show this help
  --M N         set the number of rows to N [100000]
  --N N         set the number of colums to N [0]
  --nnz N       set the number of non-zero elements to N [0]
  --repeat N    repeat N times [5]
  --format F    set sparse matrix format to F [coo]
  --algo A      set algorithm to A [serial]
  --seed S      set random seed to S [4567890123]
```

Learn how it works
=================

Compile it with -O0 -g options and run it with small parameters inside a debugger.

```
$ ... modify Makefile and set -O0 and -g to cflags ...

$ make -B
g++  -Wall -Wextra -O0 -g -fopenmp   -c -o spmv.o spmv.cc
g++ -o spmv spmv.o  -Wall -Wextra -O0 -g -fopenmp

$ gdb ./spmv

   ...
Reading symbols from ./spmv...done.

(gdb) b main
Breakpoint 1 at 0x402dc8: file spmv.cc, line 790.

(gdb) run --M 10
Starting program: /home/tau/parallel-distributed-handson/00spmv/spmv --M 10
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Breakpoint 1, main (argc=3, argv=0x7fffffffe828) at spmv.cc:790
790     int main(int argc, char ** argv) {

```

If you use Emacs, it is much better to use it from within Emacs.

```
M-x gud-gdb
```

and continue as before.


How to modify the file
=================

