c_files := $(wildcard axpy.c)
exes := $(patsubst %.c,%,$(c_files))
asms := $(patsubst %.c,%.s,$(c_files))

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

CC := gcc
CFLAGS := $(opts)
all : $(exes) $(asms)

asm : $(asms)

$(exes) : % : %.c Makefile
	$(CC) -o $@ $(CFLAGS) $<
$(asms) : %.s : %.c Makefile
	$(CC) -S $(CFLAGS) $<

clean :
	rm -f $(asms) $(exes)
