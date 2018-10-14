exe := spmv
cflags :=
cflags += -Wall -Wextra

## use debug options or optimization
#cflags += -O0 -g
cflags += -O3

## use either -fopenmp or -Wno-unknown-pragmas to supress warnings around unknown omp pragmas
cflags += -fopenmp
#cflags += -Wno-unknown-pragmas

CXX:=g++
CXXFLAGS:=$(cflags)

LDFLAGS:=$(cflags)
LIBS:=

all : $(exe)

$(exe) : % : %.o
	$(CXX) -o $@ $< $(LDFLAGS) $(LIBS)

clean :
	rm -f *.o
