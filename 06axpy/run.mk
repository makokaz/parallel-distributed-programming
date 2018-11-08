include stuff/psweep.mk

parameters := method m threads

input = out/created
output = out/out_$(method)_$(m)_$(threads).txt
cmd = OMP_NUM_THREADS=$(threads) ./axpy $(method) $(m) > $(output)

out/created :
	mkdir -p $@

# various methods
method := scalar simd simd_c simd_m simd_m_nmn simd_m_mnm simd_parallel_m_mnm
m := 1
threads := 1
$(define_rules)

# simd_c with many vars
method := simd_c simd_m_mnm
m := $(shell seq 2 15)
threads := 1
$(define_rules)

method := simd_m 
m := $(shell seq 2 40)
threads := 1
$(define_rules)


.DELETE_ON_ERROR:
