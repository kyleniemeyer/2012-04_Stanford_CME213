all : mergesort radixsort

mergesort : mergesort.cpp
        g++ -o mergesort -O3 mergesort.cpp -fopenmp

radixsort : radixsort.cpp
        g++ -o radixsort -O3 radixsort.cpp -fopenmp

clean :
        rm mergesort radixsort
