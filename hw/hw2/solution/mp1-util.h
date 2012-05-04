struct event_pair
{
  cudaEvent_t start;
  cudaEvent_t end;
};


inline void check_launch(char * kernel_name)
{
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    printf("error on %s kernel\n%s",kernel_name, cudaGetErrorString(err));
    exit(1);
  }
}


inline void start_timer(event_pair * p)
{
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}


inline void stop_timer(event_pair * p, const char * kernel_name)
{
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);
  
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  printf("%s took %.2f ms\n",kernel_name, elapsed_time);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
}

//the volatiles are needed to get correct output
//when compiling with optimization
//gcc seems to do something silly otherwise
template<typename floatType>
bool AlmostEqual2sComplement(volatile floatType A, volatile floatType B, int maxUlps)
{
    // Make sure maxUlps is non-negative and small enough that the
    // default NAN won't compare as equal to anything.
    // assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);
    if (sizeof(floatType) == 4) {
        volatile int aInt = *(volatile int*)&A;
        // Make aInt lexicographically ordered as a twos-complement int
        if (aInt < 0)
            aInt = 0x80000000 - aInt;
        // Make bInt lexicographically ordered as a twos-complement int
        volatile int bInt = *(volatile int*)&B;
        if (bInt < 0)
            bInt = 0x80000000 - bInt;
        int intDiff = abs(aInt - bInt);
        if (intDiff <= maxUlps)
            return true;
        return false;
    }
    else if (sizeof(floatType) == 8) {
        volatile long aInt = *(volatile long*)&A;
        // Make aInt lexicographically ordered as a twos-complement int
        if (aInt < 0)
            aInt = static_cast<long>(0x80000000) - aInt;
        // Make bInt lexicographically ordered as a twos-complement int
        volatile long bInt = *(volatile long*)&B;
        if (bInt < 0)
            bInt = static_cast<long>(0x80000000) - bInt;
        long intDiff = abs(aInt - bInt);
        if (intDiff <= maxUlps)
            return true;
        return false;
    }
    return false;
}
