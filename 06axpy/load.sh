#!/bin/bash

db=a.sqlite
rm -f ${db}

stuff/txt2sql ${db} --table a \
              -f 'out/out_(?P<host>[^_]+)' \
              -e 'algo = (?P<algo>.+)' \
              -e 'bs = (?P<bs>\d+)' \
              -e 'c = (?P<c>\d+)' \
              -e 'm = (?P<m>\d+)' \
              -e 'n = (?P<n>\d+)' \
              -e 'FMAs = (?P<fmas>\d+)' \
              -e '(?P<clocks_per_iter>.*?) clocks/iter, (?P<ref_clocks_per_iter>.*?) REF clocks/iter, (?P<ns_per_iter>.*?) ns/iter' \
              -r '(?P<fmas_per_clock>.*?) FMAs/clock, (?P<fmas_per_ref_clock>.*?) FMAs/REF clock, (?P<gflops>.*?) GFLOPS' \
        out/out_*.txt

sqlite3 ${db} 'select * from a limit 5'
