/* Machine Problem 4, Part 2
 * Implement a Parallel Radix Sort w/OpenMP
 * 
 * We have given you a serial radix sort implementation.
 *
 * Using the same ideas covered in class related to reduction and scan
 * to sweep values up from parallel blocks, perform a serial scan
 * and then push those values back down to parallel blocks to determine
 * the global offset for each scatter pass.
 *
 */

#include <algorithm>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include "omp.h"

typedef unsigned int uint;

void radixSortParallelHistoBlock(const std::vector<uint>::iterator blockStart, const std::vector<uint>::iterator blockEnd,
                                uint *histo, uint startBit, uint numBits)
{    
    uint mask = (1 << numBits) - 1;
    for(int i = 0; i < blockEnd - blockStart; ++i)
    {
        uint key = (*(blockStart+i) >> startBit) & mask;
        histo[key]++;
    }
}

void radixSortParallelScatterBlock(const std::vector<uint>::iterator blockStart, const std::vector<uint>::iterator blockEnd,
                                   uint *globalScan, std::vector<uint> &output, uint startBit, uint numBits)
{
    uint mask = (1 << numBits) - 1;
    for (int i = 0; i < blockEnd - blockStart; ++i) {
        uint key = (*(blockStart+i) >> startBit) & mask;
        uint globalOffset = globalScan[key]++;
        output[globalOffset] = *(blockStart+i);        
    }
}

void radixSortParallelPass(std::vector<uint> &keys, std::vector<uint> &sorted, uint numBits, uint startBit, uint blockSize)
{
    uint numBuckets = 1 << numBits;

    /* PARALLEL SECTION */
    //go over each block and compute its local histogram
    //UPSWEEP 1
    uint num_threads = (keys.size() + blockSize -1)/blockSize;
    std::vector<uint> local_histogram(num_threads*numBuckets);
    #pragma omp parallel for 
    for( int i = 0; i < num_threads; ++i)
    {
        radixSortParallelHistoBlock(keys.begin()+i*blockSize, (i==(num_threads-1))? keys.end(): keys.begin()+(i+1)*blockSize , 
                                    &local_histogram[i*numBuckets], startBit, numBits);
    }
    
    std::vector<uint> histogramRadixFrequency(numBuckets);
    /* SERIAL SECTION */
    //then reduce all the local histograms into a global one
    //UPSWEEP 2
    #pragma omp parallel for 
    for(int i = 0; i < numBuckets; ++i)
    {
        for(int j = 0; j < num_threads; ++j)
        {
            histogramRadixFrequency[i] += local_histogram[j*numBuckets + i];
        }
    }
    //now we scan this global histogram
    //SCAN
    // Although we can parallelize this part, but it's very small, 
    // and the overhead will actually show down the computation
    int sum = 0;
    for (int i = 0; i < numBuckets; ++i) {
        int x = histogramRadixFrequency[i];
        histogramRadixFrequency[i] = sum;
        sum += x;
    }
    //now we take this global scan and push the values back down
    //to the local histograms
    //DOWNSWEEP 2
    std::vector<uint> local_histogram_final(num_threads*numBuckets);
    for(int j = 0; j < numBuckets; ++j)
    {
            local_histogram_final[j] = histogramRadixFrequency[j];
    }
    for(int i = 1; i < num_threads ; ++i)
    {   
        //#pragma omp parallel for 
        for(int j = 0; j < numBuckets; ++j)
        {
            local_histogram_final[i*numBuckets + j] = local_histogram_final[(i-1)*numBuckets + j] + local_histogram[(i-1)*numBuckets + j];
        }
    }
    /* PARALLEL SECTION */
    //finally we take the local histograms and use them to compute
    //the scatter offset for each key in the block and
    //put the value in its final position
    //DOWNSWEEP 1
    #pragma omp parallel for 
    for( int i = 0; i < num_threads; ++i)
    {
        radixSortParallelScatterBlock(keys.begin()+i*blockSize, (i==(num_threads-1))? keys.end(): keys.begin()+(i+1)*blockSize , 
                                       &local_histogram_final[i*numBuckets], sorted, startBit, numBits);
    }
}

int radixSortParallel(std::vector<uint> &keys, std::vector<uint> &keys_tmp, uint numBits)
{         
    assert(numBits <= 16);
   // int n_loop = 128;
    int blockSize = 8192;//(keys.size()+n_loop-1)/n_loop; 
    std::cout<<"BlockSize:"<<blockSize<<"\n";
    for (int startBit = 0; startBit < 32; startBit += 2 * numBits) {
        radixSortParallelPass(keys,     keys_tmp, numBits, startBit,         blockSize);
        radixSortParallelPass(keys_tmp, keys,     numBits, startBit+numBits, blockSize);
    }
}

void radixSortSerialPass(std::vector<uint> &keys, std::vector<uint> &keys_radix, uint startBit, uint numBits)
{
    uint numBuckets = 1 << numBits;
    uint mask = numBuckets - 1;

    //compute the frequency histogram
    std::vector<uint> histogramRadixFrequency(numBuckets,0);
    for (int i = 0; i < keys.size(); ++i) {
        uint key = (keys[i] >> startBit) & mask;
        ++histogramRadixFrequency[key];
    }

    //now scan it
    std::vector<uint> exScanHisto(numBuckets, 0);
    for (int i = 1; i < numBuckets; ++i) {
        exScanHisto[i] = exScanHisto[i-1] + histogramRadixFrequency[i-1];
        histogramRadixFrequency[i-1] = 0;
    }

    histogramRadixFrequency[numBuckets - 1] = 0;

    //now add the local to the global and scatter the result
    for (int i = 0; i < keys.size(); ++i) {
        uint key = (keys[i] >> startBit) & mask;

        uint localOffset = histogramRadixFrequency[key]++;
        uint globalOffset = exScanHisto[key] + localOffset;

        keys_radix[globalOffset] = keys[i];
    }
}

int radixSortSerial(std::vector<uint> &keys, std::vector<uint> &keys_radix, uint numBits) {
    assert(numBits <= 16);
    for (int startBit = 0; startBit < 32; startBit += 2 * numBits) {
        radixSortSerialPass(keys,       keys_radix, startBit  ,  numBits);
        radixSortSerialPass(keys_radix, keys,       startBit+numBits, numBits);
    }
}

int main(int argc, char **argv)
{
    int n_elements = 40000000;
    int numBits = 8;
    if (argc == 2) { n_elements = atoi(argv[1]);}
    if (argc == 3) { numBits = atoi(argv[2]); n_elements = atoi(argv[1]);}
    std::cout<<"n_elements: "<<n_elements<<"\nnumBits: "<<numBits<<"\n";
    
    std::vector<uint> keys(n_elements);

    for (int i = 0; i < keys.size(); ++i) {
        keys[i] = rand();
    }

    std::vector<uint> keys_stl = keys;

    double startstl = omp_get_wtime();
    std::sort(keys_stl.begin(), keys_stl.end());
    double endstl = omp_get_wtime();

    std::cout << "stl: " << endstl - startstl << std::endl;
    
    std::vector<uint> keys_serial = keys;
    std::vector<uint> keys_pingpong(keys.size());

    //If we change the number of bits to not be an exact multiple of 32,
    //we might need to do an odd number of passes and with not all passes over
    //the same number of bits.  In that case, the final sorted result might not
    //be in keys_serial, but keys_pingpong
    double startRadixSerial = omp_get_wtime();
    radixSortSerial(keys_serial, keys_pingpong, numBits);
    double endRadixSerial = omp_get_wtime();

    for (int i = 0; i < keys.size(); ++i) {
        assert(keys_stl[i] == keys_serial[i]);
    }

    std::cout << "serial radix: " << endRadixSerial - startRadixSerial << std::endl;

    //same thing, if we do an odd number of passes, the result might be in
    //keys_pingpong, not keys
    double startRadixParallel = omp_get_wtime();
    radixSortParallel(keys, keys_pingpong, numBits);
    double endRadixParallel = omp_get_wtime();
    std::cout << "parallel radix: " << endRadixParallel - startRadixParallel << std::endl;

    for (int i = 0; i < keys.size(); ++i) {
        assert(keys_stl[i] == keys[i]);
    }


    return 0;
}
