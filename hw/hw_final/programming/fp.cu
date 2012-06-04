/* CUDA Machine Final Project
 *
 * Dong-Bang Tasi
 * May 27, 2012
 * Stanford University
 *
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h> 
#include <thrust/generate.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/inner_product.h>
#include <assert.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/generate.h>
#include <thrust/scan.h>
#include "omp.h"

template<typename floatType>
__device__
void scan_warp ( floatType * curr , const unsigned int lane, const unsigned int start, const unsigned int end )
{
    if ( lane >= 1  && start+lane < end)  curr[start+lane] = curr[start+lane-1 ] + curr[start+lane];
    if ( lane >= 2  && start+lane < end)  curr[start+lane] = curr[start+lane-2 ] + curr[start+lane]; 
    if ( lane >= 4  && start+lane < end)  curr[start+lane] = curr[start+lane-4 ] + curr[start+lane];
    if ( lane >= 8  && start+lane < end)  curr[start+lane] = curr[start+lane-8 ] + curr[start+lane];
    if ( lane >= 16 && start+lane < end)  curr[start+lane] = curr[start+lane-16] + curr[start+lane];
}

template<typename floatType>
__global__
void SegmentedScan(floatType *curr, floatType *xx, int* s, int p, int threads)
{
    int thread_id = threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x; // global thread index
    int warp_id = thread_id / threads + 1;// global warp index, from 1 to p-1
    int lane = thread_id & (threads - 1); // thread index within the warp

    if(warp_id < p)
    {
        int start = s[warp_id-1];
        int end = s[warp_id];

        while(start < end)
        {
            __syncthreads();
            scan_warp(&curr[0], lane, start, end);
            start += 31;
        }
    }
}

template<typename floatType>
struct id_op : public thrust::unary_function<floatType,floatType> 
{
    __host__ __device__
    floatType operator()(floatType x) const
    {
      return x;
    }
};


int cpu_check = 0;

int main(int argc, char **argv) {
   if(argc < 3) {
        printf("Run command: ./fp \"file a.txt\" \"file x.txt\"\n");
        exit(0);
    }
    if( argc == 4){ cpu_check = 1;}

    std::ifstream ifs_a(argv[1]);
    if (!ifs_a.good()) {
        std::cerr << "Couldn't open " << argv[1] << std::endl;
        return 1;
    }
    
    typedef float CPUFloatType;
    typedef float GPUFloatType;

    
    int n, p, q, iters;
    ifs_a >> n >> p >> q >> iters;
    thrust::host_vector<CPUFloatType> a(n); 
    thrust::host_vector<int> s(p); 
    thrust::host_vector<int> k(n); 
    for(int i=0; i<n; ++i){ ifs_a >> a[i];}
    for(int i=0; i<p; ++i){ ifs_a >> s[i];}
    for(int i=0; i<n; ++i){ ifs_a >> k[i];}
    ifs_a.close();
    std::ifstream ifs_b(argv[2]);
    if (!ifs_b.good()) {
        std::cerr << "Couldn't open " << argv[2] << std::endl;
        return 1;
    }
    thrust::host_vector<CPUFloatType> x(q); 
    for(int i=0; i<q; ++i){ ifs_b >> x[i];}
    ifs_b.close();

    std::cout<<"\nDim of a: "<<n<<"\nDim of x: "<<q<<"\nDim of s: "<< p<<"\n# of iters: "<<iters<<"\n\n";
    
    // Scan the s array, and determine the structure
    thrust::host_vector<int> key(n);
    for(int i=0,s_pos=0; i<n; ++i)
    {
        if(!( s[s_pos]<i+1 && i<s[s_pos+1])){ s_pos++;}
        key[i] = s_pos;
      //  if(i<300)std::cout<<i<<" "<<key[i]<<"\n";  
    }
    
    thrust::host_vector<CPUFloatType> cpu_buffer; 
    CPUFloatType* cpu_curr;
    CPUFloatType* cpu_prev;
    // Since x will be used several times, let's flat it to increase the memory access coalesce.
    thrust::host_vector<CPUFloatType> xx(n); 
    for(int i=0; i<n; ++i){ xx[i] = x[k[i]];}

    double cpu_start_time = 0;
    double cpu_end_time = 0;
    
    if( cpu_check != 0)
    {
        
        cpu_buffer.resize(2*n);
        thrust::copy(a.begin(), a.end(), cpu_buffer.begin());
        cpu_curr = &cpu_buffer[0];
        cpu_prev = &cpu_buffer[n];
        
        cpu_start_time = omp_get_wtime(); 
        for(int iter=0; iter<iters;++iter)
        {  
            std::swap(cpu_curr, cpu_prev);
            #pragma omp parallel for 
            for(int i=0; i<n; ++i){ cpu_prev[i] *= xx[i];}
            // Perform a segmented scan in CPU
            #pragma omp parallel for 
            for(int i=1; i<p; ++i)
            {
                cpu_curr[s[i-1]] = cpu_prev[s[i-1]];
                for(int j=s[i-1]+1; j<s[i]; ++j)
                {
                    cpu_curr[j] = cpu_curr[j-1] + cpu_prev[j];
                }
            }
        }
        cpu_end_time = omp_get_wtime();
    }  
    
    thrust::device_vector<GPUFloatType> gpu_buffer(n);
    thrust::device_vector<GPUFloatType> xx_gpu = xx;
    thrust::device_vector<int> s_gpu = s;
    thrust::device_vector<int> key_gpu = key;
    thrust::device_ptr<GPUFloatType> gpu_curr;
    gpu_curr = &gpu_buffer[0];
    thrust::copy(a.begin(), a.end(), gpu_buffer.begin());

    int threads_in_segment = 32;
    dim3 threads(512, 1, 1);
    int grid_size = (p*threads_in_segment + threads.x - 1)/threads.x; 
    //std::cout<<"grid:"<<grid_size<<"\n";
    dim3 blocks(128, (grid_size + 127)/128);

    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    for(int iter=0; iter<iters;++iter)
    { 
        thrust::transform(gpu_curr, gpu_curr+n, xx_gpu.begin(), gpu_curr, thrust::multiplies<GPUFloatType>());
        SegmentedScan<GPUFloatType><<<blocks, threads>>>(thrust::raw_pointer_cast(gpu_curr), 
                                                         thrust::raw_pointer_cast(&xx_gpu[0]), 
                                                         thrust::raw_pointer_cast(&s_gpu[0]), p, threads_in_segment);
    }
        
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);

    thrust::host_vector<GPUFloatType> gpu_result_on_host(n);
    thrust::copy(gpu_curr, gpu_curr+n, gpu_result_on_host.begin());
    
    std::cout<<"The running time of my code for "<<iters<<" iterations is: "<<elapsed_time<< " milliseconds.\n\n";

    if (cpu_check != 0) 
    {   double tol = 10e-3;
        std::cout<<"The CPU running time of my code for "<<iters<<" iterations is: "<<(cpu_end_time-cpu_start_time)*1000<< " milliseconds.\n\n";
        std::cout<<"Checking the correctness by the result from CPU\n\n";
        std::ofstream ofs_cpu("b_cpu.txt");
        for (int i = 0; i < n; ++i) {
            if( std::abs(cpu_curr[i] - gpu_result_on_host[i]) > tol)
            {
                std::cout<<"i: "<<i<<", "<<std::abs(cpu_curr[i] - gpu_result_on_host[i]) <<"\n";
              //  assert( std::abs(cpu_curr[i] - gpu_result_on_host[i]) < tol) ;
            }
            ofs_cpu << cpu_curr[i] << " ";
        }
        ofs_cpu.close();
    }

    std::ofstream ofs_gpu("b.txt");
    for (int i = 0; i < n; ++i) {
        ofs_gpu << gpu_result_on_host[i] << " ";
    }
    ofs_gpu.close();

    
    return 0;
}
