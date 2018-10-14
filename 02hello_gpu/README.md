This directory explains how to run CUDA programs.


Build
================
```
$ make
nvcc -o hello_gpu hello_gpu.cu 
```

Run (submit)
================

```
$ srun -p p --gres gpu:1 ./hello_gpu
OK
```

When submitting a job, be sure

 * you specify a partition of GPU nodes (-p p or -p v)
 * specify --gres gpu:1, which requests a GPU on the node.  If omitted, the program raises the following error.
```
$ srun -p p ./hello_gpu
hello_gpu: hello_gpu.cu:29: int main(): Assertion `a_host[i] == i' failed.
srun: error: p101: task 0: Aborted (core dumped)
```
What in fact happens is you failed to launch the kernel.  You should have checked and detected it right after you launched the kernel.  You will learn how to do this in 03spmv

 

