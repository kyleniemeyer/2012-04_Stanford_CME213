all: pagerank cipher

pagerank: pagerank.cu mp1-util.h
	nvcc -o pagerank pagerank.cu -O3 --generate-code arch=compute_20,code=sm_20 --generate-code arch=compute_11,code=sm_11 --generate-code arch=compute_10,code=sm_10 --generate-code arch=compute_12,code=sm_12 --generate-code arch=compute_13,code=sm_13

cipher: cipher.cu mp1-util.h
	nvcc -o cipher cipher.cu -O3 --generate-code arch=compute_20,code=sm_20 --generate-code arch=compute_11,code=sm_11 --generate-code arch=compute_10,code=sm_10 --generate-code arch=compute_12,code=sm_12 --generate-code arch=compute_13,code=sm_13

clean:
	rm cipher pagerank
