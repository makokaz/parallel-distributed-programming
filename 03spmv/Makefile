app := spmv

#
# CPU options
#
cxxflags :=
cxxflags += -Wall -Wextra
## use debug options or optimization
#cxxflags += -O0 -g
cxxflags += -O3
## use either -fopenmp or -Wno-unknown-pragmas to supress warnings around unknown omp pragmas
cxxflags += -fopenmp
#cxxflags += -Wno-unknown-pragmas

#
# GPU options
#
nvcxxflags :=
nvcxxflags += --gpu-code sm_60
nvcxxflags += --gpu-architecture compute_60
#nvcxxflags += --maxrregcount 64
#nvcxxflags += -O0 -Xptxas -O0,-v -g -G
nvcxxflags += -O3 -Xptxas -O3,-v

################################

CXX := g++
CXXFLAGS := $(cxxflags)
LDFLAGS :=
LIBS :=

NVCXX := /usr/local/cuda/bin/nvcc
NVCXXFLAGS := $(nvcxxflags)
NVLDFLAGS :=
NVLIBS :=

exe :=
exe += $(app).gcc
#exe += $(app).nvcc

all : $(exe)

$(app).gcc : %.gcc : %.cc
	$(CXX) -o $@ $< $(CXXFLAGS) $(LDFLAGS) $(LIBS)

$(app).nvcc : %.nvcc : %.cc
	$(NVCXX) -o $@ -x cu $< $(NVCXXFLAGS) $(NVLDFLAGS) $(NVLIBS)

clean :
	rm -f *.o
