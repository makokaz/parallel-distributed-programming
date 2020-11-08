#森下
#近藤
藤堂
#森下
#須田?
#相澤
#江崎
#相田
関谷
#千葉
川原
#関本


#
# Makefile for VGG
#

#
# compilers to compile the code with
# (comment out one if you are not interested in it)
#
cxxs :=
cxxs += g++
#cxxs += nvcc

#
# specify the full path to compilers if necessary
#
g++ := g++
nvcc := nvcc
#nvcc := /usr/local/bin/nvcc

#
# flags common across both compilers (g++ and nvcc)
#
flags :=
# ---- optimize or debug ---- 
flags += -O3
#flags += -O0 -g
# ---- turn on/off array index checking ---- 
flags += -DARRAY_INDEX_CHECK=0
#flags += -DARRAY_INDEX_CHECK=1
# ---- maximum batch size ---- 
flags += -DMAX_BATCH_SIZE=64
#flags += -DMAX_BATCH_SIZE=1
# ---- channels of the first stage ---- 
#flags += -DN_FIRST_CHANNELS=64
flags += -DN_FIRST_CHANNELS=16

#
# flags applied only to g++
#
g++flags += -Wall -Wextra
g++flags += -Wno-strict-overflow
#g++flags += -march=native

#
# flags applied only to nvcc
#
nvccflags :=
nvccflags += --generate-code arch=compute_60,code=sm_60
nvccflags += --generate-code arch=compute_70,code=sm_70
nvccflags += --compiler-options=-mavx2
#nvccflags += --maxrregcount 64
nvccflags += -Xptxas -O3,-v
#nvccflags += -Xptxas -O0,-v -G
nvccflags += -x cu

#
#
#

exes := $(addprefix vgg.,$(cxxs))

#
# you probably do not need to change below
#

targets := $(exes)

all : $(targets)

headers := $(wildcard include/*.h)

$(targets) : vgg.% : vgg.cc $(headers) Makefile
	$($*) $(flags) $($*flags) -o $@ $<

clean :
	rm -f $(targets)

clean_tag :
	rm -rf GPATH GTAGS GRTAGS HTML

tag :
	htags --suggest

.DELETE_ON_ERROR :
