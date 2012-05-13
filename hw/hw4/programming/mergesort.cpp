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
    // Run serial algorithm when the merge size is small in order to prevent the overhead.
    int *array_left_end  = array_left  + sizeleft;
    int *array_right_end = array_right + sizeright;

    if((sizeright + sizeleft) < mergeThreshold || (sizeright + sizeleft) <256)
    {
        while( array_left < array_left_end && array_right < array_right_end ) 
        {
            if( *array_left <= *array_right ) { *array_out++ = *array_left++;}
            else{ *array_out++ = *array_right++;}
        }
        while( array_left  < array_left_end  ) { *array_out++ = *array_left++;}
        while( array_right < array_right_end ) { *array_out++ = *array_right++;}
        return 0;
    }
    else
    {
        size_t left_median_indx = sizeleft/2;
        // the pointer that the right node has value higher than array_left[left_median_indx]
        size_t right_lower_indx =(std::upper_bound (&array_right[0], &array_right[sizeright], array_left[left_median_indx]) - array_right); 
        
        if(right_lower_indx < sizeright)
        {   
            #pragma omp task 
            parallel_merge(&array_left[0], left_median_indx+1, &array_right[0], right_lower_indx, &array_out[0]);
            #pragma omp task 
            parallel_merge(&array_left[left_median_indx+1], sizeleft-left_median_indx-1, &array_right[right_lower_indx], sizeright -right_lower_indx, &array_out[left_median_indx+right_lower_indx+1]);
            #pragma omp taskwait
            return 0;
        }
        else
        {//   std::cout<<"edge case\n";
          //  std::cout<<"edge, sizeleft: "<< sizeleft <<" sizeright: "<<sizeright<<"\n";
            parallel_merge(&array_left[0], left_median_indx+1, &array_right[0], sizeright, &array_out[0]);
            array_left = &array_left[left_median_indx+1];
            array_out = &array_out[left_median_indx+sizeright+1];
            while( array_left  < array_left_end  ) { *array_out++ = *array_left++;}
            return 0;
        }
    }
    return 0;
}

int merge_sort(int *array, int *array_copy, int size)
{   // Reach the bottom layer 
    if( size <=1 ){ return 1;}
    
    // Run serial algorithm when the sort size is small in order to prevent the overhead.
    if(size < sortThreshold)
    { 
        std::sort( &array[0], &array[size]);
        return 1;
    }
    int left_status, right_status;
    #pragma omp task shared(left_status)
    left_status = merge_sort( &array[0], &array_copy[0], size/2);
    #pragma omp task shared(right_status)
    right_status = merge_sort( &array[size/2] ,&array_copy[size/2] , size - size/2);
    #pragma omp taskwait
    
    // When left_status == 1 means that the left node data is in the array memory. 
    // when left_status == -1 means that the left node data is in the temp memory.
    // when left_status != right_status, it's not balance. need to copy.
    if(left_status == right_status)
    {
        if(left_status==1)
        {
            parallel_merge(&array[0], size/2, &array[size/2], size-size/2, &array_copy[0]);
            return -1;
        }
        else//if(left_status==-1)
        {
            parallel_merge(&array_copy[0], size/2, &array_copy[size/2], size-size/2, &array[0]);
            return 1;
        }
    }
    else // The left and right nodes are not in the same array. find which one is shorter, and copy that one.
    {   // The left node has more elements
        // std::cout<<"Memory Copy\n";
        if(size/2>size - size/2)
        {
            if(right_status == 1)
            {
                std::copy(&array[size/2], &array[(size/2)+size-size/2] ,&array_copy[size/2] );
                parallel_merge(&array_copy[0], size/2, &array_copy[size/2], size-size/2, &array[0]);
                return 1;
            }
            else
            {
                std::copy(&array_copy[size/2], &array_copy[(size/2)+size-size/2] ,&array[size/2] );
                parallel_merge(&array[0], size/2, &array[size/2], size-size/2, &array_copy[0]);
                return -1;
            }
        }
        else
        {
            if(left_status == 1)
            {
                std::copy(&array[0], &array[size/2] ,&array_copy[0] );
                parallel_merge(&array_copy[0], size/2, &array_copy[size/2], size-size/2, &array[0]);
                return 1;
            }
            else
            {
                std::copy(&array_copy[0], &array_copy[size/2] ,&array[0] );
                parallel_merge(&array[0], size/2, &array[size/2], size-size/2, &array_copy[0]);
                return -1;
            }
        }
    }
    return 0;
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

        std::cout << "STL sort took: " << omp_get_wtime() - startStlTime << std::endl;
    }

    double startMergeTime = omp_get_wtime();

    int status = 0;
    //TODO: call your merge sort here
    #pragma omp parallel
    #pragma omp single
    // Since I implement the ping-pong strategy, 
    // if return 1, the result will be in sort_merge, and if -1, the result will be in mergeSortTmp.
    status = merge_sort(&sort_merge[0], &mergeSortTmp[0], sort_merge.end() - sort_merge.begin() );

    std::cout << "Merge sort took: " << omp_get_wtime() - startMergeTime << std::endl;

    if (doSerialTiming) {
        for (int i = 0; i < sort_merge.size(); ++i) {
            //TODO: you will need to change this
            //if you implement ping-ponging since the result might end up
            //in mergeSortTmp
            if(status == 1) { assert(sort_stl[i] == sort_merge[i]);}
            else{ assert(sort_stl[i] == mergeSortTmp[i]); }
        }
    }
}
