#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdlib.h>
#include "omp.h"
#include "assert.h"

int sortThreshold  = 100000;
int mergeThreshold = 3000000;

int parallel_merge(int *array_left, int sizeleft, int *array_right, int sizeright, int *array_out)
{
    if (sizeleft < mergeThreshold || sizeright < mergeThreshold) {
        std::merge(array_left, array_left + sizeleft, array_right, array_right + sizeright, array_out);
    }
    else {
        //find the position in right that partitions based upon the median of left
        int midL = sizeleft / 2;
        int partitionR = std::upper_bound(array_right, array_right + sizeright, array_left[midL]) - array_right;
#pragma omp task
        parallel_merge(array_left, midL, array_right, partitionR, array_out);
#pragma omp task
        parallel_merge(array_left + midL, sizeleft - midL, array_right + partitionR, sizeright - partitionR, array_out + midL + partitionR);
#pragma omp taskwait
    }
}

int merge_sort(int *array, int *array_copy, int size, int recurse_depth)
{
    int sizeleft = size / 2 + size % 2;
    int sizeright = size / 2;
  

    if (sizeright > sortThreshold) {
        int fliparray1, fliparray2;
#pragma omp task shared(fliparray1)
       fliparray1 = merge_sort(array, array_copy, sizeleft, recurse_depth+1);
#pragma omp task shared(fliparray2)
       fliparray2 = merge_sort(array + sizeleft, array_copy + sizeleft, sizeright, recurse_depth+1);
#pragma omp taskwait
        assert(fliparray1 == fliparray2);

        if (!fliparray1) {
            parallel_merge(array, sizeleft, array + sizeleft, sizeright, array_copy);
        }
        else {
            parallel_merge(array_copy, sizeleft, array_copy + sizeleft, sizeright, array);
        }
        return fliparray1 ^ 1;
    }
    else {
        std::sort(array, array + size);
        return 0;
    }
}

int main(int argc, char **argv)
{
    if (argc != 5)
        return 1;

    sortThreshold = atoi(argv[1]);
    mergeThreshold = atoi(argv[2]);
    std::vector<int> unsorted(atoi(argv[3]));
    int doSerialTiming = atoi(argv[4]);

    for (int i = 0; i < unsorted.size(); ++i)
        unsorted[i] = rand();

    std::vector<int> sort_stl = unsorted;
    std::vector<int> sort_merge = unsorted;
    std::vector<int> mergeSortTmp(sort_merge.size());

    if (doSerialTiming) {
        double startStlTime = omp_get_wtime();
        std::sort(sort_stl.begin(), sort_stl.end());

        std::cout << "STL sort took: " << omp_get_wtime() - startStlTime << std::endl;
    }

    double startMergeTime = omp_get_wtime();

    int finalarray;
#pragma omp parallel
#pragma omp single
    finalarray = merge_sort(&sort_merge[0], &mergeSortTmp[0], sort_merge.size(), 0);

    std::cout << "Merge sort took: " << omp_get_wtime() - startMergeTime << std::endl;

    if (doSerialTiming) {
        for (int i = 0; i < sort_merge.size(); ++i) {
            if (!finalarray)
                assert(sort_stl[i] == sort_merge[i]);
            else    
                assert(sort_stl[i] == mergeSortTmp[i]);
        }
    }
}
