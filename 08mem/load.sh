#!/bin/bash
set -e
parallel2_dir=$HOME/parallel2
a2sql_dir=stuff
txt2sql=${a2sql_dir}/txt2sql

db=a.sqlite
rm -f ${db}

${txt2sql} ${db} --table a \
    -f 'output/out_.*?_(?P<cpu_node>\d+)_(?P<mem_node>\d+).txt' \
    -e '--------- (?P<rep>\d+) ---------' \
    -e 'host=(?P<host>.*)' \
    -e '(?P<n>\d+) elements x (?P<nc>\d+) chains x (?P<nscan>\d+) scans x (?P<nthreads>\d+) threads = (?P<nrecords>\d+) record accesses = (?P<nloads>\d+) loads' \
    -e 'data: (?P<sz>\d+) bytes' \
    -e 'shuffle: (?P<shuffle>.*)' \
    -e 'payload: (?P<payload>.*)' \
    -e 'stride: (?P<stride>.*)' \
    -e 'prefetch: (?P<prefetch>.*)' \
    -e 'method: (?P<method>[^\s]+)' \
    -e 'metric:l2_lines_in\.all = \d+ -> \d+ = (?P<l2_lines_in>\d+)' \
    -e 'metric:r0404 = \d+ -> \d+ = (?P<l2_lines_in>\d+)' \
    -e 'metric:r1ff1 = \d+ -> \d+ = (?P<l2_lines_in>\d+)' \
    -e '(?P<cpu_clocks>\d+) CPU clocks' \
    -e '(?P<ref_clocks>\d+) REF clocks' \
    -e '(?P<nano_sec>.*?) nano sec' \
    -e '(?P<bytes_per_clock>.*?) bytes/clock' \
    -e '(?P<gb_per_sec>.*?) GiB/sec' \
    -e '(?P<cpu_clocks_per_rec>.*?) CPU clocks per record' \
    -r '(?P<ref_clocks_per_rec>.*?) REF clocks per record' \
    output/out_*.txt



