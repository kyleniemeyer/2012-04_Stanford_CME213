//Machine Problem 4, Part 1
//Implement a recursive mergesort using OpenMP tasks

//STL algorithms you might find useful - 
//sort        - http://www.cplusplus.com/reference/algorithm/sort/
//upper_bound - http://www.cplusplus.com/reference/algorithm/upper_bound/
//merge       - http://www.cplusplus.com/reference/algorithm/merge/
//copy        - http://www.cplusplus.com/reference/algorithm/copy/

//Mergesort is inherently an out-of-place algorithm which is why we must allocate space for
//a copy of the array.  It is easiest to always use the same array for temporary space for the merging
//however this leads to a lot of unnecessary copying.  It would be fastest if you figured out how to
//"ping-pong" between them.  This means that the final answer might be in either array depending on
//how many levels of recursion happened.

//Do not change any function interfaces or names and any of the names of existing variables

//Recommended is to first just implement paralle_merge as a serial merge and get the parallel merge_sort
//working.  Then change the merge to also be parallel.

#include <vector>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include "omp.h"
#include "assert.h"

int sortThreshold  = 100000;
int mergeThreshold = 3000000;

int parallel_merge(int *array_left, int sizeleft, int *array_right, int sizeright, int *array_out)
{
}

int merge_sort(int *array, int *array_copy, int size)
{
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        std::cout << "order of arguments is: sortThreshold mergeThreshold numElementsToSort performSerialSort" << std::endl;
        std::cout << "setting performSerialSort to any non-zero value will perform std::sort and check for correctness of the merge sort" << std::endl;
        std::cout << "you will want to set this to 0 when you are doing tuning" << std::endl;
        return 1;
    }

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

        std::cout << "Stl sort took: " << omp_get_wtime() - startStlTime << std::endl;
    }

    double startMergeTime = omp_get_wtime();

    //TODO: call your merge sort here

    std::cout << "Merge sort took: " << omp_get_wtime() - startMergeTime << std::endl;

    if (doSerialTiming) {
        for (int i = 0; i < sort_merge.size(); ++i) {
            //TODO: you will need to change this
            //if you implement ping-ponging since the result might end up
            //in mergeSortTmp
            assert(sort_stl[i] == sort_merge[i]);
        }
    }
}
