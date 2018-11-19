Hands-on Report on SpMV
=========================

********************************
Name: ***YOUR NAME***
Student ID: ***YOUR STUDENT ID***
Your repository: https://doss-gitlab.eidos.ic.i.u-tokyo.ac.jp/***YOUR_NAME***/parallel-distributed-handson
********************************

How to use this file
====================

> * This file gives you a template of your hands-on report
> * You edit this file in place, commit and push the change, and send me a notification through ITC-LMS (go to this lecture's page in ITC-LMS)
> * For hands-on reports, I do not expect you to write a detailed explanation and analysis on what you have done.  It should mostly take you only to write the code, execute the exercise and copy-paste the log in this text file.
> * Instructions are somewhat detailed on Exercise 1.  Subsequent exercises will follow the same procedure.
> * Below, in parts marked as
>```
>  ********************************************************
>  *** copy paste your command line and the output here ***
>  ********************************************************
>```
>you are supposed to copy and paste what you did on the command line and what you saw in the terminal.
> * Commented parts (lines starting with '>') are instructions that should not be part of your report.  Delete them after you think you understand the instruction.
> * Description about each exercise problem is ../index.html in this repository, or https://www.eidos.ic.i.u-tokyo.ac.jp/~tau/lecture/parallel_distributed/2018/handson/tau/parallel-distributed-handson/03spmv/index.html in public.

Exercise 1
=================

> Your task: Parallelize spmv for coo format with OpenMP "parallel for" and run it on multicore/many core CPUs.

Execution log that shows your code successfully compiles
----------------------------------------------------
```
  ********************************************************
  *** copy paste your command line and the output here ***
  ********************************************************
```

Execution log that shows your code successfully runs
----------------------------------------------------

> It needs to include at least the following.

 * run the parallel algorithm with a single OpenMP thread. e.g.
```
srun -p big -t 0:05:00 "OMP_NUM_THREADS=1 ./spmv.gcc -a parallel"
```
  
```
  ********************************************************
  *** copy paste your command line and the output here ***
  ********************************************************
```
 * run the parallel algorithm with multiple OpenMP threads, say 10.
```
srun -p big -t 0:05:00 -n 1 -c 10 bash -c "OMP_NUM_THREADS=10 OMP_PROC_BIND=true ./spmv.gcc -a parallel"
```

> Consult ../01hello/README.md for the meaning of options -n 1 and -c 10 of srun.

> Since the command line tends to be long and tedious to type, I recommend you to put them in a shell script. e.g., write a shell script "run.sh" as follows
> ```
> #!/bin/bash -x
> OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OMP_PROC_BIND=true ./spmv.gcc -a parallel
> ```
>   - put -x so that it prints executed commands
>   - then run them as follows.
>
> ```
> $ chmod +x run.sh         # if you haven't done so
> $ srun -p big -t 0:05:00 -n 1 -c 10 ./run.sh
> ```


```
  ********************************************************
  *** copy paste your command line and the output here ***
  ********************************************************
```

>  * try the same procedure for other matrices
>   * extremely thin matrices; e.g. --M 10 --N 50000000 or even --M 1
>   * extremely tall matrices; e.g. --M 50000000 --N 10 or even --N 1
>   * "-t one" option for deterministic result. specifically,
>     * choose a number, say 1234, and call it L
>     * choose two numbers both larger than L; call them M and N
>     * give -t one --M M --N N --nnz $((L * L))
>     * then the result (the value of lambda) MUST BE exactly (L * L). e.g.,
> ```   
> $ srun -p big -t 0:05:00 -n 1 -c 10 bash -c "OMP_NUM_THREADS=10 OMP_PROC_BIND=true ./spmv.gcc -a parallel -t one --M 2000 --N 3000 --nnz $((1234 * 1234))"
>    ...
> lambda = 1.522756000e+06
> $ echo $((1234 * 1234))
> ```

```
  ********************************************************
  *** copy paste your command line and the output here ***
  ********************************************************
```

Execution log that shows your code successfully improves performance (at least somewhat)
----------------------------------------------------

> Several of you observed that your parallel version does not perform any better than the serial version and asked why.  At this point, I did not explain why it might be slower but depending on parameter settings, it is not surprising.  Recall that multiple threads may update the same element of y[i].  We resolve the race condition by making the accumulation atomic, which is more expensive than ordinary load and store.  More important, though, is that if two threads update the same or even near elements (e.g., y[100] and y[101]), it will involve an expensive cache miss as I will explain in the lecture soon.  That is, even if it were ordinary loads/stores, it tends to be more expensive when you do it with multiple threads.
>
> For now, you simply try to find parameters (--M, --N and --nnz) that make it less likely that multiple threads update the same y[i] at the same time, for which you basically make the number of zeros per row and per column small compared to the number of threads.  In an extreme case in which there is only a single nnz in a row or a column, there is no way that multiple threads update the same element.  There are still possibilities that multiple threads update near elements, but in general, if nnz/M and nnz/N are very small, it is less likely that an update is expensive.
>
> The other (much better) strategy is to arrange non-zero elements so that elements update by threads do not (or are unlikely to) overlap.  This is exactly what coo_sorted or csr format accomplishes.
>
> Putting them together, try the following.

>  * make M and N large; make nnz just as large as them (i.e., make nnz/M nearly one). e.g.,
>  * use -f coo_sorted option
>  * altogether, try the following settings 
> ```
> ./spmv.gcc -a serial   --M 10000000 --N 10000000 --nnz 10000000 -f coo
> ./spmv.gcc -a serial   --M 10000000 --N 10000000 --nnz 10000000 -f coo_sorted
> OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OMP_PROC_BIND=true ./spmv.gcc -a parallel --M 10000000 --N 10000000 --nnz 10000000 -f coo
> OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OMP_PROC_BIND=true ./spmv.gcc -a parallel --M 10000000 --N 10000000 --nnz 10000000 -f coo_sorted
> ./spmv.gcc -a serial   --M 100000   --N 100000   --nnz 10000000 -f coo
> ./spmv.gcc -a serial   --M 100000   --N 100000   --nnz 10000000 -f coo_sorted
> OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OMP_PROC_BIND=true ./spmv.gcc -a parallel --M 100000   --N 100000   --nnz 10000000 -f coo
> OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OMP_PROC_BIND=true ./spmv.gcc -a parallel --M 100000   --N 100000   --nnz 10000000 -f coo_sorted
> ```
>  * and throw the job as follows, changing THE_NUMBER_OF_THREADS_YOU_WANT_TO_USE from 2 to whichever number you like.  
> ```
> srun -p big -t 0:05:00 -n 1 -c THE_NUMBER_OF_THREADS_YOU_WANT_TO_USE ./run.sh
> ```
>  * the primary performance metric you need to watch is the GFLOPS number in the  line like
> ```
>  201500000 flops in 0.172431 sec (1.168581 GFLOPS)
> ```
> Try to get this number as high as possible.

```
  ********************************************************
  *** copy paste your command line and the output here ***
  ********************************************************
```


Exercise 2
=================

> Your task: Parallelize spmv for coo format with CUDA. Follow the instruction below.

> Do something similar to Exercise 2.  I do not write detailed instructions below.

```
  ********************************************************
  *** copy paste your command line and the output here ***
  ********************************************************
```

Exercise 3
=================

> Your task: Parallelize spmv for csr format with OpenMP "parallel for" and run it on multicore/many core CPUs. Read and follow the details below.

> Do something similar to Exercise 3

```
  ********************************************************
  *** copy paste your command line and the output here ***
  ********************************************************
```

Exercise 4
=================

> Your task: Parallelize spmv for csr format with CUDA. Read and follow the details below.

> Do something similar to Exercise 4

```
  ********************************************************
  *** copy paste your command line and the output here ***
  ********************************************************
```

Optional Exercise 5
===================

> Your task: Parallelize spmv for csr format with tasks. Read and follow the details below.

> Do something similar to Exercise 5

```
  ********************************************************
  *** copy paste your command line and the output here ***
  ********************************************************
```
Optional Exercise 6
===================

> Your task: Parallelize spmv for coo_sorted format with parallel for + user-defined reductions. Read and follow the details below.

> Do something similar to Exercise 5

```
  ********************************************************
  *** copy paste your command line and the output here ***
  ********************************************************
```

Optional Exercise 7
=================

> Your task: analyze an achievable GFLOPS (upper bound), coming from the memory bandwidth constraint.

> Write your analysis

Challenging Optional Exercise 8
=================

> Your task: do whatever you think of to approach the achievable GFLOPS you obtained in Exercise 7.

> Write whatever you thought, did and saw.
