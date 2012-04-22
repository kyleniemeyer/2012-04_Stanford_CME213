all: 2dHeat

2dHeat: 2dHeat.cu mp1-util.h
	nvcc -o 2dHeat 2dHeat.cu -O3 -arch=sm_20

clean:
	rm 2dHeat
