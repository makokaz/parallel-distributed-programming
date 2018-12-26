headers := $(wildcard include/*.h)
unit_check_exes := $(patsubst include/%.h,include/%.chk,$(headers))

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

all : $(targets)

vgg.gcc : vgg.cc $(headers) Makefile
	$(CXX) $(CXXFLAGS) $(g++FLAGS) -o $@ vgg.cc

vgg.nvcc : vgg.cc $(headers) Makefile
	$(NVCXX) $(CXXFLAGS) $(nvcxxFLAGS) -o $@ vgg.cc

unit_check : $(unit_check_exes)

$(unit_check_exes) : include/%.chk : include/%.h
	g++ $(CXXFLAGS) -o $@ include/unit_check.cc -DINC_H=\"$*.h\" -D$*_main=main

clean :
	rm -f vgg.gcc vgg.nvcc include/*.chk
