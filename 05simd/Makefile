#CC := gcc
CC := clang

common_flags := 
common_flags += -Wall
common_flags += -Wextra
common_flags += -O3

gcc_flags :=
gcc_flags += -fopenmp-simd
gcc_flags += -fno-tree-slp-vectorize
gcc_flags += -fopt-info-vec-optimized
gcc_flags += -fopt-info-vec-missed
gcc_flags += -mavx512f
gcc_flags += -mfma

clang_flags :=
clang_flags += -fopenmp-simd
clang_flags += -fno-slp-vectorize 
clang_flags += -Rpass=vectorize
clang_flags += -Rpass-missed=vectorize
clang_flags += -Rpass-analysis=vectorize
clang_flags += -mavx512f
clang_flags += -mfma

icc_flags :=
icc_flags += -fopt-report-phase=vectorize
icc_flags += -fopt-report-phase-missed=vectorize

CFLAGS := $(common_flags) $($(CC)_flags)

c := $(wildcard *.c)
exe := $(patsubst %.c,%,$(c))
asm := $(patsubst %.c,%.s,$(c))

all : $(asm)

$(asm) : %.s : %.c
	$(CC) $(CFLAGS) -S $<

clean :
	rm -f $(asm) $(exe)

