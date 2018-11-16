#!/usr/bin/python
import sys,os,types

# ------------- preamble -------------

import smart_gnuplotter

def Es(s):
    sys.stderr.write(s)

def get_unique(g, db, f):
    return g.do_sql(db,
                    '''
select distinct %s from a 
order by %s
''' % (f, f))

def get_max(g, db, f):
    return g.do_sql(db,
                    '''
select max(%s) from a 
''' % f)

g = smart_gnuplotter.smart_gnuplotter()
#g.default_terminal = 'epslatex color size 9cm,6cm font "" 8'
#g.default_terminal = 'svg'
#g.default_terminal = 'svg'
#g.default_terminal = 'emf color solid font ",18"'

sqlite_file = sys.argv[1] if len(sys.argv) > 1 else "a.sqlite"
out_dir     = sys.argv[2] if len(sys.argv) > 2 else "graphs"

db = g.open_sql(sqlite_file)


# ------------- contents -------------

def mk_plot_title(b):
    if b["eq"] == "=":
        return "local"
    else:
        return "remote"

# -------------- latency with 1 chain --------------

def graph_latency():
    # show latency of link list traversal
    # x : size of the data
    # y : latency per access
    # (1) only local
    # (2) compare local and remote
    for eqs,conf,label in [ ([ "=" ],      "local", "local"), 
                            ([ "=", "<>" ],"local_remote", "local and remote") ]:
        output = "%s/latency_%s_%%(min_sz)s" % (out_dir, conf)
        g.graphs((db,
                  '''
select sz,avg(cpu_clocks_per_rec) 
from a 
where method="ptrchase"
  and nc=1
  and nthreads=1
  and sz>=%(min_sz)s
  and shuffle=1
  and prefetch=0
  and payload=1
  and cpu_node %(eq)s mem_node
group by sz 
order by sz;
''',
                  "","",[]),
                 #output=output,
                 #terminal="svg",
                 graph_vars=[ "min_sz" ],
                 graph_title=("latency per load in a list traversal (%s) [$\\\\geq$ %%(min_sz)s]" % label),
                 graph_attr='''
set logscale x
#set xtics rotate by -20
set key left
#unset key
''',
                 yrange="[0:]",
                 ylabel="latency/load",
                 xlabel="size of the region (bytes)",
                 plot_with="linespoints",
                 plot_title=mk_plot_title,
                 eq=eqs,
                 min_sz=[ 0, 10 ** 8 ],
                 verbose_sql=2,
                 save_gpl=0)

# -------------- l2 miss --------------
def graph_cache(event):
    # show latency of link list traversal
    # x : size of the data
    # y : latency per access
    # (1) only local
    # (2) compare local and remote
    g.graphs((db,
              '''
              select 
              sz,
              avg(%(event)s/(nloads+0.0)),
              cimin(%(event)s/(nloads+0.0),0.001),
              cimax(%(event)s/(nloads+0.0),0.001)
from a 
where method="ptrchase"
  and nc=1
  and nthreads=1
  and sz>=%(min_sz)s
  and shuffle=1
  and prefetch=0
  and payload=0
group by sz 
order by sz;
''',
                  "","",[]),
             #output = "%s/cache_miss_%%(min_sz)s" % out_dir,
             #terminal="svg",
             graph_vars=[ "min_sz" ],
             graph_title="cache miss rate of a list traversal [$\\\\geq$ %(min_sz)s]",
             graph_attr='''
#set logscale x
#set xtics rotate by -20
set key left
#unset key
''',
             yrange="[0:]",
             ylabel="miss rate",
             xlabel="size of the region (bytes)",
             plot_with="yerrorlines",
             plot_title="",
             min_sz=[ 0 ],
             event=[event],
             verbose_sql=2,
             save_gpl=0)

# -------------- bandwidth local vs remote --------------
def graph_bw_ptrchase():
    for eqs,conf,label in [ ([ "=" ], "local", "local"), 
                            ([ "=", "<>" ], "local_remote", "local and remote") ]:
        output = "%s/bw_%s_%%(min_sz)s" % (out_dir, conf)
        g.graphs((db,
                  '''
select sz,avg(gb_per_sec) 
from a 
where method="ptrchase"
  and nc=1
  and nthreads=1
  and sz>=%(min_sz)s
  and shuffle=1
  and prefetch=0
  and payload=1
  and cpu_node %(eq)s mem_node
group by sz 
order by sz
''',
                  "","",[]),
                 output=output,
                 graph_vars=[ "min_sz" ],
                 graph_title=("bandwidth (%s) [$\\\\geq$ %%(min_sz)s]" % label),
                 graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
                 yrange="[0:]",
                 ylabel="bandwidth (GB/sec)",
                 xlabel="size of the region (bytes)",
                 plot_with="linespoints",
                 plot_title=mk_plot_title,
                 eq=eqs,
                 min_sz=[ 0, 10 ** 8 ],
                 verbose_sql=2,
                 save_gpl=0)

# -------------- bandwidth with X chains --------------
def graph_bw_ptrchase_chains():
    for eqs,conf in [ ([ "="  ], "local"), 
                      ([ "<>" ], "remote") ]:
        output = "%s/bw_chains_%s_%%(min_sz)s" % (out_dir, conf)
        g.graphs((db,
                  '''
select sz,avg(gb_per_sec) 
from a 
where method="ptrchase"
  and nc=%(nc)s
  and nthreads=1
  and sz>=%(min_sz)s
  and shuffle=1
  and prefetch=0
  and payload=1
  and cpu_node %(eq)s mem_node
group by sz 
order by sz
''',
                  "","",[]),
                 output=output,
                 graph_vars=[ "min_sz" ],
                 graph_title=("bandwidth with a number of chains [%s, $\\\\geq$ %%(min_sz)s]" % conf),
                 graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
                 yrange="[0:]",
                 ylabel="bandwidth (GB/sec)",
                 xlabel="size of the region (bytes)",
                 plot_with="linespoints",
                 plot_title="%(nc)s chains",
                 nc=get_unique(g, db, "nc"),
                 eq=eqs,
                 min_sz=[ 0, 10 ** 8 ],
                 verbose_sql=2,
                 save_gpl=0)

# -------------- bandwidth with X chains --------------
def graph_bw_prefetch():
    output = "%s/bw_prefetch_%%(min_sz)s" % out_dir
    g.graphs((db,
              '''
select sz,avg(gb_per_sec) 
from a 
where method="ptrchase"
  and nc=1
  and nthreads=1
  and sz>=%(min_sz)s
  and shuffle=1
  and prefetch=%(prefetch)s
  and payload=1
  and cpu_node=mem_node
group by sz 
order by sz
''',
                  "","",[]),
                 output=output,
                 graph_vars=[ "min_sz" ],
                 graph_title="bandwidth with a number of chains [$\\\\geq$ %(min_sz)s]",
                 graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
                 yrange="[0:]",
                 ylabel="bandwidth (GB/sec)",
                 xlabel="size of the region (bytes)",
                 plot_with="linespoints",
                 plot_title="prefetch=%(prefetch)s",
                 prefetch=[0,10],
                 min_sz=[ 0, 10 ** 8 ],
                 verbose_sql=2,
                 save_gpl=0)
    
def graph_methods():
    # compare link list traversal vs. random index
    output = "%s/methods_%%(min_sz)s" % out_dir
    g.graphs((db,
              '''
select sz,avg(gb_per_sec) 
from a 
where method = "%(method)s"
  and nc=1
  and nthreads=1
  and sz>=%(min_sz)s
  and shuffle=1
  and prefetch=0
  and payload=1
  and cpu_node=mem_node
group by sz 
order by sz
''',
              "","",[]),
             output=output,
             graph_vars=[ "min_sz" ],
             graph_title="bandwidth of random list traversal vs random array traversal [$\\\\geq$ %(min_sz)s]",
             graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
             yrange="[0:]",
             ylabel="bandwidth (GB/sec)",
             xlabel="size of the region (bytes)",
             plot_with="linespoints",
             plot_title="%(method)s",
             method=get_unique(g, db, "method"),
             min_sz=[ 0, 10 ** 8 ],
             verbose_sql=2,
             save_gpl=0)

# -------------- bandwidth with X threads --------------
def graph_bw_ptrchase_threads():
    # show the effect of increasing number of threads with max chains
    output = "%s/bw_threads_%%(min_sz)s" % out_dir
    g.graphs((db,
              '''
select sz,avg(gb_per_sec) 
from a 
where method="ptrchase"
  and nc=%(nc)s
  and nthreads=%(nthreads)s
  and sz>=%(min_sz)s
  and shuffle=1
  and payload=1
  and cpu_node=mem_node
group by sz 
order by sz
''',
                  "","",[]),
             output=output,
             graph_vars=[ "min_sz" ],
             graph_title="bandwidth with a number of threads [$\\\\geq$ %(min_sz)s]",
             graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
             yrange="[0:]",
             ylabel="bandwidth (GB/sec)",
             xlabel="size of the region (bytes)",
             plot_with="linespoints",
             plot_title="%(nc)s chains, %(nthreads)s threads",
             #nc=get_unique(g, db, "nc"),
             nc=[1,10],
             #nthreads=get_unique(g, db, "nthreads"),
             nthreads=[1,4,16],
             min_sz=[ 0, 10 ** 8 ],
             verbose_sql=2,
             save_gpl=0)

def mk_plot_title_prefetch(b):
    if b["shuffle"] == 0:
        return "address-sorted list"
    else:
        return "random list"

def graph_sort_vs_unsorted():
    # compare two link list traversals
    # randomly ordered list vs address-sorted list
    output = "%s/sorted_%%(min_sz)s" % out_dir
    g.graphs((db,
              '''
select sz,avg(gb_per_sec) 
from a 
where method="ptrchase"
  and nc=1
  and nthreads=1
  and sz>=%(min_sz)s
  and shuffle=%(shuffle)s
  and prefetch=0
  and payload=1
  and cpu_node=mem_node
group by sz 
order by sz
''',
              "","",[]),
             output=output,
             graph_vars=[ "min_sz" ],
             graph_title="bandwidth of random list traversal vs address-ordered list traversal [$\\\\geq$ %(min_sz)s]",
             graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
             yrange="[0:]",
             ylabel="bandwidth (GB/sec)",
             xlabel="size of the region (bytes)",
             plot_with="linespoints",
             plot_title=mk_plot_title_prefetch,
             shuffle=[ 0, 1 ],
             min_sz=[ 0, 10 ** 8 ],
             verbose_sql=2,
             save_gpl=0)


def mk_plot_title_all_access(b):
    method   = b["method"]
    shuffle  = ("" if b["shuffle"]  else " (sorted list)")
    prefetch = (" (prefetch)" if b["prefetch"] else "")
    nc       = ((" (%d chains)" % b["nc"]) if b["nc"] > 1 else "")
    if method == "ptrchase":
        return "%s%s%s%s" % (method, shuffle, prefetch, nc)
    else:
        return "%s" % method

def graph_summary():
    # seq vs random vs ptrchase
    output = "%s/summary_%%(min_sz)s" % out_dir
    g.graphs((db,
              '''
select sz,avg(gb_per_sec) 
from a 
where method="%(method)s"
  and nc=%(nc)s
  and nthreads=1
  and sz>=%(min_sz)s
  and shuffle=%(shuffle)s
  and prefetch=%(prefetch)s
  and payload=1
  and cpu_node=mem_node
group by sz 
order by sz
''',
              "","",[]),
             output=output,
             # terminal = 'epslatex color size 12cm,6cm font "" 8',
             graph_vars=[ "min_sz" ],
             graph_title="bandwidth of various access patterns [$\\\\geq$ %(min_sz)s]",
             graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
             yrange="[0:]",
             ylabel="bandwidth (GB/sec)",
             xlabel="size of the region (bytes)",
             plot_with="linespoints",
             plot_title=mk_plot_title_all_access,
             method=[ "ptrchase", "random", "sequential" ],
             shuffle=[ 0, 1 ],
             prefetch=[ 0, 10 ],
             nc=[ 1, 10 ],
             min_sz=[ 0, 10 ** 8 ],
             verbose_sql=2,
             save_gpl=0)

if 1:
    graph_cache("l2_lines_in")
if 0:
    graph_latency()
    graph_bw_ptrchase()
    graph_bw_ptrchase_chains()
    graph_bw_prefetch()
    graph_methods()
    graph_bw_ptrchase_threads()
    graph_sort_vs_unsorted()
    graph_summary()

