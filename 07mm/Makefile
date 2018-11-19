mmc_srcs := $(wildcard mmc??.h)
idx := $(patsubst mmc%.h,%,$(mmc_srcs))
exes := $(addprefix mmc,$(idx))
asms := $(addsuffix .S,$(exes))

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
#opts += -O0 -g
opts += -Wall -Wextra

CXX := g++
CC := gcc
CXXFLAGS := $(opts)
CFLAGS := $(opts)

all : $(all)

$(exes) : mmc% : mmc.cc mmc%.h
	$(CXX) -o $@ -DMMC_H=\"mmc$*.h\" $(CXXFLAGS) $<

$(asms) : mmc%.S : mmc.cc mmc%.h
	$(CXX) -o $@ -DMMC_H=\"mmc$*.h\" $(CXXFLAGS) -S $<

$(exes) $(asms) : Makefile mmc.h

clean :
	rm -f $(exes) $(objs) $(asms)


