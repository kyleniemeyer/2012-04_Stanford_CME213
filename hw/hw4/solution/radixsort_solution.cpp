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
    for (std::vector<uint>::iterator it = blockStart; it != blockEnd; ++it) {
        uint key = ((*it) >> startBit) & mask;
        ++histo[key];
    }
}

void radixSortParallelScatterBlock(const std::vector<uint>::iterator blockStart, const std::vector<uint>::iterator blockEnd,
                                   uint *globalScan, std::vector<uint> &output, uint startBit, uint numBits)
{
    uint numBuckets = 1 << numBits;
    uint mask = numBuckets - 1;
    std::vector<uint> localHisto(numBuckets, 0);

    for (std::vector<uint>::iterator it = blockStart; it != blockEnd; ++it) {
        uint key = ((*it) >> startBit) & mask;
        uint blockOffset = globalScan[key];
        uint localOffset = localHisto[key];
        localHisto[key] += 1;

        output[blockOffset + localOffset] = *it;
    }
}

void radixSortParallelPass(std::vector<uint> &keys, std::vector<uint> &sorted, uint numBits, uint startBit, uint blockSize)
{
    uint numBuckets = 1 << numBits;
    uint numBlocks = (keys.size() + blockSize - 1) / blockSize;
    std::vector<uint> blockHistograms(numBlocks * numBuckets, 0);

    //go over each block and compute its local histogram
#pragma omp parallel for
    for(uint block = 0; block < numBlocks; ++block) {
        std::vector<uint>::iterator begin = keys.begin() + block* blockSize;
        std::vector<uint>::iterator end;
        if ( (block + 1) * blockSize >= keys.size())
            end = keys.end();
        else
            end = keys.begin() + (block + 1) * blockSize;

        radixSortParallelHistoBlock(begin, end,
                                   &blockHistograms[block * numBuckets], startBit, numBits);
    }

    //first reduce all the local histograms into a global one
    std::vector<uint> globalHisto(numBuckets, 0);
    for (uint block = 0; block < numBlocks; ++block) {
        for (uint bucket = 0; bucket < numBuckets; ++bucket) {
            globalHisto[bucket] += blockHistograms[block * numBuckets + bucket];
        }
    }

    //now we scan this global histogram
    std::vector<uint> globalHistoExScan(numBuckets, 0);
    for (uint bucket = 1; bucket < numBuckets; ++bucket) {
        globalHistoExScan[bucket] = globalHistoExScan[bucket - 1] + globalHisto[bucket - 1];
    }

    //now we do a local histogram in each block and add in the 
    //global value to get global position
    std::vector<uint> blockExScan(numBuckets * numBlocks, 0);
    std::vector<uint> runningHisto(numBuckets, 0);
    for (uint block = 0; block < numBlocks; ++block) {
        for (uint bucket = 0; bucket < numBuckets; ++bucket) {
            blockExScan[block * numBuckets + bucket] = runningHisto[bucket] + globalHistoExScan[bucket];
            runningHisto[bucket] += blockHistograms[block * numBuckets + bucket];
        }
    }

#pragma omp parallel for
    for (uint block = 0; block < numBlocks; ++block) {
        uint keyIndex = block * blockSize;
        std::vector<uint>::iterator begin = keys.begin() + block* blockSize;
        std::vector<uint>::iterator end;
        if ( (block + 1) * blockSize >= keys.size())
            end = keys.end();
        else
            end = keys.begin() + (block + 1) * blockSize;
        radixSortParallelScatterBlock(begin, end, 
                                      &blockExScan[block * numBuckets], sorted, startBit, numBits);
    }
}

int radixSortParallel(std::vector<uint> &keys, std::vector<uint> &keys_tmp, uint numBits) {
    for (int startBit = 0; startBit < 32; startBit += 2 * numBits) {
        radixSortParallelPass(keys, keys_tmp, numBits, startBit, keys.size()/8); 
        radixSortParallelPass(keys_tmp, keys, numBits, startBit + numBits, keys.size()/4); 
    }
}

void radixSortSerialPass(std::vector<uint> &keys, std::vector<uint> &keys_radix, uint startBit, uint numBits)
{
    uint numBuckets = 1 << numBits;
    uint mask = numBuckets - 1;

    //compute the frequency histogram
    std::vector<uint> histogramRadixFrequency(numBuckets);
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


int main(void)
{
    std::vector<uint> keys(40000000);

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
    radixSortSerial(keys_serial, keys_pingpong, 16);
    double endRadixSerial = omp_get_wtime();

    for (int i = 0; i < keys.size(); ++i) {
        assert(keys_stl[i] == keys_serial[i]);
    }

    std::cout << "serial radix: " << endRadixSerial - startRadixSerial << std::endl;

    //same thing, if we do an odd number of passes, the result might be in
    //keys_pingpong, not keys
    double startRadixParallel = omp_get_wtime();
    radixSortParallel(keys, keys_pingpong, 16);
    double endRadixParallel = omp_get_wtime();

    for (int i = 0; i < keys.size(); ++i) {
        assert(keys_stl[i] == keys[i]);
    }

    std::cout << "parallel radix: " << endRadixParallel - startRadixParallel << std::endl;

    return 0;
}
