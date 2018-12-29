headers := $(wildcard include/*.h)

CXX := g++
NVCXX := /usr/local/bin/nvcc

CXXFLAGS :=
CXXFLAGS += -O0 -g
#CXXFLAGS += -O3
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -Wno-strict-overflow
#CXXFLAGS += -DARRAY_INDEX_CHECK=0
CXXFLAGS += -DARRAY_INDEX_CHECK=1
CXXFLAGS += -DMAX_BATCH_SIZE=64

g++FLAGS := 
nvccFLAGS := 

targets := 
targets += vgg.gcc
#targets += vgg.nvcc

all : $(targets) $(unit_check_exes)

vgg.gcc : vgg.cc $(headers) Makefile
	$(CXX) $(CXXFLAGS) $(g++FLAGS) -o $@ vgg.cc

vgg.nvcc : vgg.cc $(headers) Makefile
	$(NVCXX) $(CXXFLAGS) $(nvcxxFLAGS) -o $@ vgg.cc

clean :
	rm -f vgg.gcc vgg.nvcc
