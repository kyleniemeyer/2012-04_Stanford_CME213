
ifdef DEBUG
	NVCCFLAGS=-arch=sm_20 -g -G
else
	NVCCFLAGS=-O3 -arch=sm_20
endif


all: create_cipher solve_cipher

create_cipher: create_cipher.cu
	nvcc -o create_cipher create_cipher.cu $(NVCCFLAGS)
solve_cipher: solve_cipher.cu
	nvcc -o solve_cipher solve_cipher.cu $(NVCCFLAGS)

clean:
	rm create_cipher solve_cipher
