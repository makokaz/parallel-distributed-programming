cc_files := $(wildcard axpy.cc)
exes := $(patsubst %.cc,%,$(cc_files))
asms := $(patsubst %.cc,%.s,$(cc_files))

opts := 
opts += -Wall
opts += -Wextra
opts += -O3
#opts += -O0 -g

# GCC
opts += -fopenmp
opts += -mavx512f
#opts += -fopt-info-vec-optimized 
#opts += -fopt-info-vec-missed
#opts += -fcilkplus
#opts += -funroll-loops
#opts += --param max-unroll-times=2

# ICC
#opts += -qopenmp
#opts += -axMIC-AVX512
#opts += -qopt-report=4
#opts += -qopt-report-phase=vec

CC := g++
#CC := icpc
CFLAGS := $(opts)
all : $(exes) $(asms)

asm : $(asms)

$(exes) : % : %.cc Makefile
	$(CC) -o $@ $(CFLAGS) $<
$(asms) : %.s : %.cc Makefile
	$(CC) -S $(CFLAGS) $<

clean :
	rm -f $(asms) $(exes)
