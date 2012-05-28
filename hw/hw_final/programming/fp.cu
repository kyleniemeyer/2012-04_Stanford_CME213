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
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h> 
#include <thrust/generate.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/generate.h>


int main(int argc, char **argv) {
   if(argc < 3) {
        printf("Run command: ./fp \"file a.txt\" \"file x.txt\"\n");
        exit(0);
    }
    int cpu_check = 0;
    cpu_check = atoi(argv[3]);

    std::ifstream ifs_a(argv[1]);
    if (!ifs_a.good()) {
        std::cerr << "Couldn't open " << argv[1] << std::endl;
        return 1;
    }
    int n, p, q, iters;
    ifs_a >> n >> p >> q >> iters;
    thrust::host_vector<float> a(n); 
    thrust::host_vector<float> s(p); 
    thrust::host_vector<float> k(n); 
    for(int i=0; i<n; ++i){ ifs_a >> a[i];}
    for(int i=0; i<p; ++i){ ifs_a >> s[i];}
    for(int i=0; i<n; ++i){ ifs_a >> k[i];}
    ifs_a.close();
    std::ifstream ifs_b(argv[2]);
    if (!ifs_b.good()) {
        std::cerr << "Couldn't open " << argv[2] << std::endl;
        return 1;
    }
    thrust::host_vector<float> x(q); 
    for(int i=0; i<q; ++i){ ifs_b >> x[i];}
    ifs_b.close();

    std::cout<<"\nDim of a: "<<n<<"\nDim of x: "<<q<<"\nDim of s: "<< p<<"\n# of iters: "<<iters<<"\n\n";

    thrust::host_vector<float> cpu_buffer; 
    float* cpu_curr;
    float* cpu_prev;
    if( cpu_check != 0)
    {
        cpu_buffer.resize(2*n);
        for(int i=0;i<n;++i){ cpu_buffer[i] = a[i];}
        cpu_curr = &cpu_buffer[0];
        cpu_prev = &cpu_buffer[n];
        for(int iter=0; iter<iters;++iter)
        { 
            std::cout<<"CPU_iter: "<<iter<<"\n";
            std::swap(cpu_curr, cpu_prev);
            int s_pos = 0;
            for(int i=0; i<n; ++i)
            {
                double sum = 0;
                while(!( s[s_pos]<i+1 && i<s[s_pos+1])){ s_pos++;}
                //if( i%1000==0) std::cout<<"  ith: "<<i <<" "<< n<<" s_pos: "<<s_pos<<" s[s_pos]: "<<s[s_pos]<<" s[s_pos+1]: "<<s[s_pos+1]<<"\n";
                for( int j=s[s_pos]; j<i+1;++j){ sum += cpu_prev[j]*x[k[j]];}
                cpu_curr[i] = sum;
            }
        }
    }  

    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);




    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);

    if (cpu_check != 0) 
    {
        std::cout<<"Checking the correctness by using the CPU\n\n";
        std::ofstream ofs_cpu("b_cpu.txt");
        for (int i = 0; i < n; ++i) {
            //assert(cpu_b[i] == gpu_b[i]);
            ofs_cpu << cpu_curr[i] << " ";
        }
        ofs_cpu.close();
    }

    std::cout<<"The running time of my code for "<<iters<<" iterations is: "<<elapsed_time<< " milliseconds.\n\n";
    
    return 0;
}
