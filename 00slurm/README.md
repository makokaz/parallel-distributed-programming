00slurm
===================

Become familiar with slurm job manager

For impatients
===================

Try the following and check if they work.

Run a command
-------------------

```
$ srun -p p   -t 0:01:00 hostname
$ srun -p big -t 0:01:00 hostname
$ srun -p knm -t 0:01:00 hostname
```

Replace "hostname" part with a command you like to use.

Run an interactive shell
-------------------

```
$ srun -p p   -t 0:01:00 --pty bash
$ srun -p big -t 0:01:00 --pty bash
$ srun -p knm -t 0:01:00 --pty bash
```

You will see the shell prompt of a compute node.
Run a few commands and quit immediately.
This is useful when you want to interactively diagnose problems.

Run a debugger
-------------------

```
$ srun -p p   -t 0:01:00 --pty cuda-gdb
$ srun -p big -t 0:01:00 --pty gdb
$ srun -p knm -t 0:01:00 --pty gdb
```

When you need to debug CPU programs, you can simply do it on the login node.
When you need to debug GPU programs, use --pty option to run the debugger (cuda-gdb) on a compute node.
It is recommended to do this from within Emacs (M-x gud-gdb).

Reservation
-------------------

The following should work during hands-on session.

```
$ srun -p p   -t 0:01:00 --reservation ptau hostname
$ srun -p big -t 0:01:00 --reservation bigtau hostname
$ srun -p knm -t 0:01:00 --reservation knmtau hostname
```

See below for more details.


srun 
===================

The most basic command is srun, which simply runs any command on a specified partition

```
$ srun -p <partition> -t <time_limit>  <command>
```

Example:
```
$ srun -p p -t 0:01:00 hostname
p101
```

This runs "hostname" command on the "p" partition of the system with execution time limit of 1 minute.

Available partitions
===================

The argument to -p option specifies a partition, which is one of:

 * p : NVIDIA P100 (12 nodes, each having 4 GPUs)
 * v : NVIDIA V100 (3 nodes, each having 8 GPUs)
 * big : Intel Skylake (2 nodes, each having 4 sockets x 16 cores x 2 virtual cores, 3TB memory)
 * knm : Intel Knights Mill (8 nodes, each having 68 cores x 4 virtual cores)

Time limit
===================

The argument to -t option is the time limit.  The scheduler tends to give jobs of short time limit more chance to run.  It is recommended to give a short time limit to jobs that you know will finish quickly.  T

The syntax is HH:MM:SS, so 0:01:00 means one minute, 1:00:00 an hour, and so on.

If unspecified, it means 48 hours.

Advanced reservation
===================

On Monday 15:00 - 18:00, I will reserve a node from each of partitions p, big and knm, to make sure students of this class can use it during hands-on sessions.  To use the reserved nodes, give --reservation {ptau,bigtau,knmtau} depending on which partition you want to use.

Example:

```
$ srun -p p --reservation ptau -t 0:01:00 hostname
p101
```

Outside the reservation period, you will see the following message.

```
$ srun -p p --reservation ptau -t 0:01:00 hostname
srun: Requested reservation not usable now
srun: job 12207 queued and waiting for resources
```

scontrol to see the info about advanced reservations
===================

You can check the information about reservations by scontrol show reservation.
In partiular, you can check if you are enlisted in the reservation.
If you don't find your login ID in the Users part, you are not.
This is simply because you do not have an account (or did not have one when I made the reservations).

Let me know your login ID when you actually have one.


```
$ scontrol show reservation
ReservationName=maintenance_1026 StartTime=2018-10-26T17:00:00 EndTime=2019-10-26T17:00:00 Duration=365-00:00:00
   Nodes=big[000-001],knm[000-007],p[100-111],v[100-102] NodeCnt=25 CoreCnt=1116 Features=(null) PartitionName=(null) Flags=MAINT,IGNORE_JOBS,SPEC_NODES,ALL_NODES
   TRES=cpu=3320
   Users=root Accounts=(null) Licenses=(null) State=INACTIVE BurstBuffer=(null) Watts=n/a

ReservationName=knmtau StartTime=2018-10-15T15:00:00 EndTime=2018-10-15T20:00:00 Duration=05:00:00
   Nodes=knm000 NodeCnt=1 CoreCnt=68 Features=(null) PartitionName=knm Flags=
   TRES=cpu=272
   Users=tau,ictish,qiao,u00059,u00060,u00062,u00063,u00064,u00066,u00067,u00069,u00070,u00071,u00072,u00073,u00075,u00076,u00077,u00078,u00079,u00080,u00081,u00083,u00084,u00086,u00087,u00088,u00089 Accounts=(null) Licenses=(null) State=INACTIVE BurstBuffer=(null) Watts=n/a

ReservationName=bigtau StartTime=2018-10-15T15:00:00 EndTime=2018-10-15T20:00:00 Duration=05:00:00
   Nodes=big000 NodeCnt=1 CoreCnt=64 Features=(null) PartitionName=big Flags=
   TRES=cpu=128
   Users=tau,ictish,qiao,u00059,u00060,u00062,u00063,u00064,u00066,u00067,u00069,u00070,u00071,u00072,u00073,u00075,u00076,u00077,u00078,u00079,u00080,u00081,u00083,u00084,u00086,u00087,u00088,u00089 Accounts=(null) Licenses=(null) State=INACTIVE BurstBuffer=(null) Watts=n/a

ReservationName=ptau StartTime=2018-10-15T15:00:00 EndTime=2018-10-15T20:00:00 Duration=05:00:00
   Nodes=p100 NodeCnt=1 CoreCnt=28 Features=(null) PartitionName=p Flags=
   TRES=cpu=56
   Users=tau,ictish,qiao,u00059,u00060,u00062,u00063,u00064,u00066,u00067,u00069,u00070,u00071,u00072,u00073,u00075,u00076,u00077,u00078,u00079,u00080,u00081,u00083,u00084,u00086,u00087,u00088,u00089 Accounts=(null) Licenses=(null) State=INACTIVE BurstBuffer=(null) Watts=n/a
```

squeue to see queued jobs
===================

```
$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             12176         v ist_run_   ictish PD       0:00      1 (Resources)
             12177         v ist_run_   ictish PD       0:00      1 (Priority)
             12147         p ist_run_   ictish  R    9:33:28      1 p103
             12148         p ist_run.   ictish  R    9:18:33      1 p102
             ...
```

The important column is ST (Status) column.

 * R : running
 * PD : pending

sinfo to see node availability
===================

```
$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
p            up 2-00:00:00      2  down* p[104-105]
p            up 2-00:00:00      7    mix p[101-103,106-109]
p            up 2-00:00:00      3   idle p[100,110-111]
v            up 2-00:00:00      3    mix v[100-102]
knm          up 2-00:00:00      8   idle knm[000-007]
big          up 2-00:00:00      1    mix big001
big          up 2-00:00:00      1   idle big000
```

More info
===================

 * consult manual pages (e.g., man srun, man squeue, etc.) on the login node or the web
 * consult IST cluster user guide https://login000.cluster.i.u-tokyo.ac.jp/wordpress/
 
