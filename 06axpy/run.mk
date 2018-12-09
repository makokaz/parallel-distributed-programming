include stuff/psweep.mk

parameters := host exe algo m n c bs threads

input = out/created
output = out/out_$(host)_$(algo)_$(m)_$(c)_$(bs)_$(threads).txt
cmd = OMP_NUM_THREADS=$(threads) ./$(exe) -a $(algo) -m $(m) -n $(n) -c $(c) --cuda-block-size $(bs) > $(output)

out/created :
	mkdir -p $@

# various algos
host:=$(shell hostname | tr -d [0-9])
n:=1000000

# single variable
exe := axpy.g++
algo := scalar simd simd_c simd_m simd_m_nmn simd_m_mnm simd_parallel_m_mnm
m := 1
c := 1
bs := 1
threads := 1
#$(define_rules)

# simd_c with many vars
exe := axpy.g++
algo := simd_c
m := 0
c := $(shell seq 2 15)
bs := 1
threads := 1
#$(define_rules)

# simd_m 
exe := axpy.g++
algo := simd_m
m := $(shell seq 16 16 320)
c := 1
bs := 1
threads := 1
#$(define_rules)

# simd_m_mnm
exe := axpy.g++
algo := simd_m_mnm
m := $(shell seq 16 16 512)
c := $(shell seq 1 16)
bs := 1
threads := 1
#$(define_rules)

# cuda cuda_c
exe := axpy.nvcc
algo := cuda cuda_c
m := 1
c := 1
bs := 1
threads := 1
$(define_rules)

# cuda_c
exe := axpy.nvcc
algo := cuda_c
m := 0
c := $(shell seq 1 1 10)
bs := $(shell seq 32 32 640)
threads := 1
#$(define_rules)


.DELETE_ON_ERROR:
