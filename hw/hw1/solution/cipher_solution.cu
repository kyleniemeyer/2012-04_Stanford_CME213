/* This is machine problem 1, part 1, shift cypher
 *
 * The problem is to take in a string (a vector of characters) and a shift amount,
 * and add that number to each element of
 * the string, effectively "shifting" each element in the 
 * string.
 * 
 * We do this in four different ways:
 * 1. With a standard cuda kernel loading chars and outputting chars for each thread
 * 2. With a standard cuda kernel, casting the character pointer to an int so that
 *    we load and store 4 bytes each time instead of 1 which gives us better coalescing
 *    and uses the memory effectively to achieve higher bandwidth
 * 3. Same spiel except with a uint2, so that we load 8 bytes each time
 * 4. Similar to #1 except that we use thrust and don't write a kernel ourselves
 *
 */


#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <fstream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>

#include "mp1-util.h"


// Repeating from the tutorial, just in case you haven't looked at it.

// "kernels" or __global__ functions are the entry points to code that executes on the GPU
// The keyword __global__ indicates to the compiler that this function is a GPU entry point.
// __global__ functions must return void, and may only be called or "launched" from code that
// executes on the CPU.

void host_shift_cypher(std::vector<unsigned char> &input_array, std::vector<unsigned char> &output_array, unsigned char shift_amount)
{
  for(unsigned int i=0;i<input_array.size();i++)
  {
    unsigned char element = input_array[i];
    output_array[i] = element + shift_amount;
  }
}

// This kernel implements a per element shift
// by naively loading one byte and shifting it
__global__ void shift_cypher(unsigned char *input_array, unsigned char *output_array, unsigned char shift_amount,  unsigned int array_length)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < array_length)
        output_array[tid] = input_array[tid] + shift_amount;
}

//Here we load 4 bytes at a time instead of just 1
//to improve the bandwidth due to a better memory
//access pattern
__global__ void shift_cypher_int(unsigned int *input_array, unsigned int *output_array, unsigned int shift_amount, unsigned int array_length) 
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < array_length)
        output_array[tid] = input_array[tid] + shift_amount;
}

//Here we go even further and load 8 bytes
//does it make a further improvement?
__global__ void shift_cypher_int2(uint2 *input_array, uint2 *output_array, unsigned int shift_amount, unsigned int array_length) 
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < array_length) {
        output_array[tid].x = input_array[tid].x + shift_amount;
        output_array[tid].y = input_array[tid].y + shift_amount;
    }
}

bool checkResults(std::vector<unsigned char> &cipher_text_host, unsigned char *device_output_array,
                  const char *type) {
    //allocate space on host for gpu results
    std::vector<unsigned char> cipher_text_from_gpu(cipher_text_host.size());

    event_pair timer;
    start_timer(&timer);
    // download and inspect the result on the host:
    cudaMemcpy(&cipher_text_from_gpu[0], device_output_array, cipher_text_host.size(), cudaMemcpyDeviceToHost);
    check_launch("copy from gpu");
    stop_timer(&timer,"copy from gpu");
  
    // check CUDA output versus reference output
    int error = 0;
    for(int i=0;i<cipher_text_host.size();i++)
    {
        if(cipher_text_host[i] != cipher_text_from_gpu[i]) 
        { 
            error = 1;
            printf("Error at pos: %d\nexpected: %d got: %d\n", i, (int)cipher_text_host[i], (int)cipher_text_from_gpu[i]);
            break;
        }
    }

    if(error)
    {
        printf("Output of CUDA %s version and host version didn't match! \n", type);
        return false;
    }

    return true;
}

int main(void)
{

  //First load the text 
  std::ifstream ifs("mobydick.txt", std::ios::binary);
  if (!ifs.good()) {
      std::cerr << "Couldn't open book file!" << std::endl;
      return 1;
  }

  std::vector<unsigned char> text;

  ifs.seekg(0, std::ios::end); //seek to end of file
  int length = ifs.tellg();    //get distance from beginning
  ifs.seekg(0, std::ios::beg); //move back to beginning

  text.resize(length);
  ifs.read((char *)&text[0], length);

  ifs.close();

  //need to make a couple copies of the book, otherwise everything happens too quickly
  //make 2^4 = 16 copies
  for (int i = 0; i < 4; ++i) {
      text.insert(text.end(), text.begin(), text.end());
  }

  // compute the size of the arrays in bytes
  // with enough padding that a uint2 access won't be out of bounds
  int num_bytes = (text.size() + 7) * sizeof(unsigned char);

  //allocate host arrays
  std::vector<unsigned char> cipher_text_gpu(text.size());
  std::vector<unsigned char> cipher_text_host(text.size());

  // pointers to device arrays
  unsigned char *device_input_array  = 0;
  unsigned char *device_output_array = 0;
  
  event_pair timer;
  
  // cudaMalloc device arrays
  cudaMalloc((void**)&device_input_array,  num_bytes);
  cudaMalloc((void**)&device_output_array, num_bytes);
  
  // if either memory allocation failed, report an error message
  if(device_input_array == 0 || device_output_array == 0)
  {
    printf("couldn't allocate memory\n");
    return 1;
  }

  // generate random input string
  unsigned char shift_amount = (rand() % 25) + 1; //we don't want the shift to be 0!
  
  // do copies to and from gpu once to get rid of timing weirdness
  // on first time accesses due to driver
  // touch all memory
  cudaMemcpy(device_input_array,  &text[0],            num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_output_array, device_input_array,  num_bytes, cudaMemcpyDeviceToDevice);
  cudaMemcpy(&text[0],            device_output_array, num_bytes, cudaMemcpyDeviceToHost);

  start_timer(&timer);
  // copy input to GPU
  cudaMemcpy(device_input_array, &text[0], num_bytes, cudaMemcpyHostToDevice);
  check_launch("copy to gpu");
  stop_timer(&timer,"copy to gpu");
  
  // generate reference output
  {
      start_timer(&timer);
      host_shift_cypher(text, cipher_text_host, shift_amount);
      stop_timer(&timer,"host shift cypher");
  }

  // choose a number of threads per block
  // we use 512 threads here
  const int block_size = 512;

  bool noErrors = true;
  // generate GPU char output
  {
      int grid_size = (text.size() + block_size - 1) / block_size;
      start_timer(&timer);
      // launch kernel
      shift_cypher<<<grid_size,block_size>>>(device_input_array, device_output_array, shift_amount, text.size());
      check_launch("gpu shift cypher char");
      stop_timer(&timer,"gpu shift cypher char");
      if (!checkResults(cipher_text_host, device_output_array, "char")) {
          noErrors = false;
      }
  }

  // generate GPU uint output
  {
      int grid_size = ( (text.size() + 3)/4 + block_size - 1) / block_size;
      unsigned int iShift = (shift_amount | (shift_amount << 8) | (shift_amount << 16) | (shift_amount << 24));
      start_timer(&timer);
      // launch kernel
      shift_cypher_int<<<grid_size,block_size>>>((unsigned int *)device_input_array, (unsigned int *)device_output_array, iShift, (text.size() + 3)/4);
      check_launch("gpu shift cypher uint");
      stop_timer(&timer,"gpu shift cypher uint");
      if (!checkResults(cipher_text_host, device_output_array, "uint")) {
          noErrors = false;
      }
  }

  //generate GPU uint2 output
  {
      int grid_size = ( (text.size() + 7)/8 + block_size - 1) / block_size;
      unsigned int iShift = (shift_amount | (shift_amount << 8) | (shift_amount << 16) | (shift_amount << 24));
      start_timer(&timer);
      // launch kernel
      shift_cypher_int2<<<grid_size,block_size>>>((uint2 *)device_input_array, (uint2 *)device_output_array, iShift, (text.size() + 7)/8);
      check_launch("gpu shift cypher uint2");
      stop_timer(&timer,"gpu shift cypher uint2");
      if (!checkResults(cipher_text_host, device_output_array, "uint2")) {
          noErrors = false;
      }
  }

  //generate GPU output with thrust
  {
      thrust::device_vector<unsigned char> dText = text;
      thrust::device_vector<unsigned char> dOutput(dText.size());
      start_timer(&timer);
      thrust::transform(dText.begin(), dText.end(), thrust::make_constant_iterator(shift_amount), dOutput.begin(), thrust::plus<unsigned char>());
      //don't need to check the launch because thrust does that for us
      stop_timer(&timer,"gpu shift cypher thrust");
      if (!checkResults(cipher_text_host, thrust::raw_pointer_cast(&dOutput[0]), "thrust")) {
          noErrors = false;
      }
  }

  if (noErrors) {
      printf("All CUDA Versions matched reference output.  Outputting ciphered text.\n");
      std::ofstream ofs("mobydick_enciphered.txt");
      //use the original length, before we made copies
      for (int i = 0; i < length; ++i) {
          ofs << cipher_text_host[i];
      }
      ofs.close();
  }
 
  // deallocate memory
  cudaFree(device_input_array);
  cudaFree(device_output_array);
}

