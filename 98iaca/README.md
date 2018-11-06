
This directory contains Intel Architecture Code Analyzer (IACA) version 3.0 for Linux.
You can download it for other OS (Windows and Mac) from the original site.
https://software.intel.com/en-us/articles/intel-architecture-code-analyzer

 * iaca-lin64/  IACA (got from the above site)
 * manual/      IACA manual (got from the above site)
 * include/     two small include files I made to make things easy

IACA is a useful tool to analyze the throughput of instruction sequences.

include/ contains small include files to make it ease to use.  Here is how.

 1. gcc -O3 (other options) -S xxx.c -o xxx.S
 2. open xxx.S and enclose the section of interest (typically a loop) with #include "begin.h" and #include "end.h", like this

#include "iaca_begin.h"
.L123
   instructions
   instructions
   instructions
   instructions
   jne .L123
#include "iaca_end.h"

 3. gcc -c xxx.S
 4. iaca xxx.o

You might need to copy iaca_begin.h and iaca_end.h into your working directory or add an appropriate -I option to the compiler in step 3.  You might also need to specify the full path to iaca (which you can find at iaca-lin64/iaca).
