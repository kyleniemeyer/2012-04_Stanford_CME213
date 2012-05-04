all: 2dHeat

2dHeat: 2dHeat_solution.cu mp1-util.h
	nvcc -o 2dHeat 2dHeat_solution.cu -O3 -arch=sm_20

clean:
	rm 2dHeat
