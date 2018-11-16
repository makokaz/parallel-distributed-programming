
include stuff/psweep.mk

numactl:=numactl -iall --

# ---------------------------------------

out_dir:=output

# ----- parameter and command definitions -----

p:=$(shell seq 8 1 23)
#p:=$(shell seq 8 1 8)
powers:=$(foreach i,$(p),$(shell echo $$((1 << $(i)))))
#n:=$(foreach p,$(powers),$(foreach o,-2 0 2 4 6,$(shell echo $$(($(p) * (9 + $(o)) / 9)))))
n:=$(shell seq 1000 200 8000) $(shell seq 10000 500 30000)

parameters:=host try method n n_chains n_threads shuffle payload cpu_node mem_node prefetch

# knm : r0404
# skx : l2_lines_in.all

cmd=(echo host=$(host) ; OMP_NUM_THREADS=$(n_threads) EV=$(EV) numactl -N $(cpu_node) -i $(mem_node) -- ./mem -m $(method) -n $(n) -c $(n_chains) -x $(shuffle) -l $(payload) -p $(prefetch) -r 7) > $(output)
input=$(out_dir)/created
output=$(out_dir)/out_$(host)_$(method)_$(n)_$(n_chains)_$(n_threads)_$(shuffle)_$(payload)_$(cpu_node)_$(mem_node)_$(prefetch)_$(try).txt

## common parameters ##
host:=$(shell hostname | tr -d [0-9])
cpu_node:=0
payload:=0
#payload:=1
try:=0 1

## effect of number of chains ##
method:=p
n_chains:=1 2 4 8 10 12 14
n_threads:=1
shuffle:=1
prefetch:=0
#mem_node:=0 1
mem_node:=0
#$(define_rules)

## effect of working set size ##
method:=p
n_chains:=1
n_threads:=1
shuffle:=1
prefetch:=0
#mem_node:=0 1
mem_node:=0
$(define_rules)

## effect of access methods ##
method:=p s r
n_chains:=1 2 4
n_threads:=1
shuffle:=1
prefetch:=0
mem_node:=0
#$(define_rules)

## effect of prefetch ##
method:=p
n_chains:=1 2 4
n_threads:=1
shuffle:=1
prefetch:=0 10
mem_node:=0
#$(define_rules)

## effect of sorted addresses ##
method:=p
n_chains:=1 2 4
n_threads:=1
shuffle:=0 1
prefetch:=0
mem_node:=0
#$(define_rules)

## many threads with pointers ##
method:=p
n_chains:=1 5 10
n_threads:=1 2 4 6 8 12 16
shuffle:=1
prefetch:=0
mem_node:=0
#$(define_rules)

## many threads with pointers ##
method:=s r
n_chains:=1
n_threads:=1 2 4 6 8 12 16
shuffle:=1
prefetch:=0
mem_node:=0
#$(define_rules)

$(out_dir)/created :
	mkdir -p $@

.DELETE_ON_ERROR:


