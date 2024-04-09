# Compilers and commands
CC=		gcc
CXX=		gcc
NVCC=		nvcc

#Flags
CFLAGS		= -W -Wall
CXXFLAGS	= -W -Wall
NVCCFLAGS	= -g -G --use_fast_math

INCPATH		= /usr/include/

all: matrix float double

matrix: matrix.c
	$(CC) -o matrix matrix.c $(CFLAGS)

float: float.cu
	$(NVCC) -o float float.cu $(NVCCFLAGS) -I$(INCPATH)
double: double.cu
	$(NVCC) -o double double.cu $(NVCCFLAGS) -I$(INCPATH)

clean:
	rm -f matrix float double

