cc_srcs := $(wildcard *.cc)
idx := $(patsubst mm%.cc,%,$(cc_srcs))
exes := $(addprefix mm,$(idx))
objs := $(addsuffix .o,$(exes)) mm.o
asms := $(addsuffix .s,$(exes))

all :=
all += $(exes)
all += $(asms)

opts := 
opts += -march=native 
#opts += -mavx512f
#opts += -mavx2
#opts += -funroll-loops 
#opts += --param max-unroll-times=100
#opts += -fopenmp-simd
#opts += -fopt-info-vec-optimized 
#opts += -fopt-info-vec-missed
#opts += -axMIC-AVX512
#opts += -mkl
opts += -O3
opts += -Wall -Wextra

CXX := g++
CC := gcc
CXXFLAGS := $(opts)
CFLAGS := $(opts)

all : $(all)

$(exes) : mm% : mm.o mm%.o 
	$(CXX) -o $@ $^

$(objs) : %.o : %.cc
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(asms) : %.s : %.cc
	$(CXX) $(CXXFLAGS) -o $@ -S $<

$(objs) $(asms) : Makefile

clean :
	rm -f $(exes) $(objs) $(asms)


