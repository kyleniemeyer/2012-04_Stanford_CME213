
ifdef DEBUG
	NVCCFLAGS=-arch=sm_20 -g -Xcompiler -fopenmp 
else
	NVCCFLAGS=-O3 -arch=sm_20 -Xcompiler -fopenmp
endif


all: fp

fp: fp.cu
	nvcc -o fp fp.cu $(NVCCFLAGS)

clean:
	rm fp
