
Code: VGG
==================

 * VGG: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
 * published at ICLR 2015 https://www.iclr.cc/archive/www/doku.php%3Fid=iclr2015:main.html

Dataset: CIFAR-10
==================

 * on IST Cluster, it's in /home/tau/cifar10
 * original source: http://www.cs.toronto.edu/~kriz/cifar.html
  * download "CIFAR-10 binary version (suitable for C programs)" if you download yourself

Compile: 
==================

```
$ make -B
g++  -O0 -g -Wall -Wextra -Wno-strict-overflow -DARRAY_INDEX_CHECK=1 -DMAX_BATCH_SIZE=64  -o vgg.gcc vgg.cc
```

(make sure you have -O3 or -O0 -g in the command line, depending on whether you want to measure performance or debug)

Run: 
==================

 * Make sure you have cifar-10-batches-bin/data_batch_1.bin file under the current directory.  On IST cluster, you can do so by
```
$ ln -s /home/tau/cifar10/cifar-10-batches-bin
```

 * and
```
$ ./vgg.gcc
```
