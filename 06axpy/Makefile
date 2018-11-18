CXX := g++
#CXX := icpc
#CXX := clang
#CXX := nvcc

cc_files := $(wildcard axpy.cc)
exes := $(patsubst %.cc,%,$(cc_files))
asms := $(patsubst %.cc,%.S,$(cc_files))

common_opts := 
common_opts += -Wall
common_opts += -Wextra
common_opts += -O3
#common_opts += -O0 -g

# GCC
g++_opts :=
g++_opts += -fopenmp
g++_opts += -march=native
#g++_opts += -mavx512f

clang_opts :=
clang_opts += -fopenmp
clang_opts += -march=native
#clang_opts += -mavx512f

# ICC
icc_opts :=
icc_opts += -qopenmp
icc_opts += -xMIC-AVX512

# NVCC
nvcc_opts :=
nvcc_opts += -x cu

CXXFLAGS := $(common_opts) $($(CXX)_opts)
all : $(exes) $(asms)

asm : $(asms)

$(exes) : % : %.cc Makefile
	$(CXX) -o $@ $(CXXFLAGS) $<
$(asms) : %.S : %.cc Makefile
	$(CXX) -o $@ -S $(CXXFLAGS) $<

clean :
	rm -f $(asms) $(exes)
