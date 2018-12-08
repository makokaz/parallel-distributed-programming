include stuff/psweep.mk

parameters := host algo m c bs threads

input = out/created
output = out/out_$(host)_$(algo)_$(m)_$(c)_$(bs)_$(threads).txt
cmd = OMP_NUM_THREADS=$(threads) ./axpy -a $(algo) -m $(m) -c $(c) --cuda-block-size $(bs) > $(output)

out/created :
	mkdir -p $@

# various algos
algo := scalar simd simd_c simd_m simd_m_nmn simd_m_mnm simd_parallel_m_mnm cuda cuda_c
host:=$(shell hostname | tr -d [0-9])

# single variable
m := 1
c := 1
bs := 1
threads := 1
#$(define_rules)

# simd_c with many vars
algo := simd_c simd_m_mnm
c := $(shell seq 2 15)
m := 0
bs := 1
threads := 1
#$(define_rules)

# simd_m 
algo := simd_m
c := 1
m := $(shell seq 32 16 640)
bs := 1
threads := 1
#$(define_rules)

# cuda_c
algo := cuda_c
c := $(shell seq 1 1 10)
bs := $(shell seq 32 32 640)
m := 0
threads := 1
$(define_rules)


.DELETE_ON_ERROR:
