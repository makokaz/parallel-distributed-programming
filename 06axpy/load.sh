#!/bin/bash

db=a.sqlite
rm -f ${db}

stuff/txt2sql ${db} --table a \
        -e 'algo = (?P<algo>.+)' \
        -e 'm = (?P<m>\d+)' \
        -e 'n = (?P<n>\d+)' \
        -e 'flops = (?P<flops>\d+)' \
        -e '(?P<cpu_clocks_per_iter>.*?) CPU clocks/iter, (?P<ref_clocks_per_iter>.*?) REF clocks/iter, (?P<ns_per_iter>.*?) ns/iter' \
        -r '(?P<flops_per_cpu_clock>.*?) flops/CPU clock, (?P<flops_per_ref_clock>.*?) flops/REF clock, (?P<gflops>.*?) GFLOPS' \
        out/out_*.txt

sqlite3 ${db} 'select count(*) from a'
