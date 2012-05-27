
ifdef DEBUG
	NVCCFLAGS=-arch=sm_20 -g 
else
	NVCCFLAGS=-O3 -arch=sm_20
endif


all: fp

fp: fp.cu
	nvcc -o fp fp.cu $(NVCCFLAGS)

clean:
	rm fp
