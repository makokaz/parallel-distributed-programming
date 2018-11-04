cc_files := $(wildcard axpy.cc)
exes := $(patsubst %.cc,%,$(cc_files))
asms := $(patsubst %.cc,%.s,$(cc_files))

opts := 
opts += -Wall
opts += -Wextra
opts += -march=native 
opts += -fopenmp
opts += -O3
#opts += -O0 -g
#opts += -fopt-info-vec-optimized 
#opts += -fopt-info-vec-missed
#opts += -std=gnu99
#opts += -fopenmp
#opts += -fcilkplus
#opts += -funroll-loops
#opts += --param max-unroll-times=2

CC := g++
CFLAGS := $(opts)
all : $(exes) $(asms)

asm : $(asms)

$(exes) : % : %.cc Makefile
	$(CC) -o $@ $(CFLAGS) $<
$(asms) : %.s : %.cc Makefile
	$(CC) -S $(CFLAGS) $<

clean :
	rm -f $(asms) $(exes)
