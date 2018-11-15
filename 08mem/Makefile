cc_files := $(wildcard *.cc)
exe := $(patsubst %.cc,%,$(cc_files))
asm := $(addsuffix .s,$(exe))

opts := 
opts += -O3
opts += -Wall -Wextra
opts += -march=native
opts += -funroll-loops
#opts += -fopenmp

libs :=
libs += -lpfm

CXX := g++
CXXFLAGS := $(opts)
LIBS := $(libs)

all : $(exe) $(asm)

$(exe) : % : %.cc event.h
	$(CXX) -o $@ $(CXXFLAGS) $< $(LIBS)

$(asm) : %.s : %.cc event.h
	$(CXX) -o $@ -S $(CXXFLAGS) $<

clean :
	rm -f mem

