/* CUDA Machine Problem 3 - Thrust
 *
 * Vigenere Ciphers
 * Part 1 - Encoding
 *
 * Your Assignment:
 * Fill in the TODOs to create a program that will take a plain text in,
 * reduce it to only lowercase letters and then encode it with a Vigenere cipher.
 * You are only allowed to use thrust functions and iterators. The functors you will
 * need have been defined, you need to fill them in.  As usual don't change the
 * interface of any existing functions or any variable names.
 *
 * Useful Reference Info:
 * ASCII A-Z [65, 90]
 * ASCII a-z [97, 122]
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

struct isnot_lowercase_alpha : thrust::unary_function<bool, unsigned char>{
    //TODO: fill in this functional
    __host__ __device__
    bool operator()(const unsigned char &c)
    {
        return ( ('a' > c) || ( 'z' < c) );
    }
};

struct upper_to_lower : thrust::unary_function<unsigned char, unsigned char>{
    //TODO: fill in this functional
    __host__ __device__
    unsigned char operator()(const unsigned char &c)
    {
        // char upper_a = 'A';
        // char lower_a = uppera | 0x20;
        // Note that lower_a == lower_a | 0x20
        return c | 0x20;
    }
};

//This functional has to be initialized with the period and (a pointer) to the table of
//shifts.  You will need a constructor.
struct apply_shift : thrust::binary_function<unsigned char, unsigned int, unsigned char> {
    //TODO: fill in the functional
    __host__ __device__
    unsigned char operator()(const unsigned char &c, const unsigned int &shift)
    {
        return ((c - 'a') + shift) % 26 + 'a';
    }
};

struct periodic_shifts_fun : thrust::unary_function<unsigned int, size_t> 
{ 
    const unsigned int period; 
    unsigned int * shifts;
    periodic_shifts_fun(const unsigned int period, unsigned int * shifts) : period(period), shifts(shifts){} 
    __host__ __device__ 
    unsigned int operator()(const size_t i) 
    { 
        return shifts[i % period]; 
    } 
}; 


int main(int argc, char **argv) {
   if(argc < 3) {
        printf("Run command: ./create_cipher input.txt period\n");
        exit(0);
    }

    std::ifstream ifs(argv[1], std::ios::binary);
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

    unsigned int period = atoi(argv[2]);

    //sanitize input to contain only a-z lowercase
    thrust::device_vector<unsigned char> dText = text;
    thrust::device_vector<unsigned char> plain_text(text.size());
    
    //TODO: With one thrust call, generate the cleaned output text which only has lowercase letters
    //all spaces, etc. removed and uppercase letters converted to lowercase
    //this result should end up in plain_text
    
    //TODO: make sure this gets set to the right value
    //the number of characters in the cleaned output
    int numElements = thrust::remove_copy_if(  thrust::make_transform_iterator(dText.begin(), upper_to_lower()), 
                             thrust::make_transform_iterator(dText.end(),   upper_to_lower()), 
                             plain_text.begin(), isnot_lowercase_alpha()) - plain_text.begin();
    
    thrust::device_vector<unsigned int> shifts(period);

    //TODO: Use thrust's random number generation capability to initialize the shift vector
    // Create a minstd_rand object to act as our source of randomness
    thrust::minstd_rand rng;
    // The key should be lowercase letter
    thrust::uniform_int_distribution<unsigned int> dist(1,26); 
    for(int i=0;i<period;++i){ shifts[i] = dist(rng);}

    thrust::device_vector<unsigned char> device_cipher_text(numElements);

    //TODO: Again, with one thrust call, create the cipher text from the plaintext     
    thrust::transform_iterator<periodic_shifts_fun, thrust::counting_iterator<size_t>  > 
        periodic_shifts_iter = thrust::make_transform_iterator(  thrust::make_counting_iterator((size_t)0),
                                                                 periodic_shifts_fun(period, thrust::raw_pointer_cast(&shifts[0]))); 
                    
    thrust::transform(  plain_text.begin(), 
                        plain_text.begin() + numElements, 
                        periodic_shifts_iter,
                        device_cipher_text.begin(), apply_shift());

    thrust::host_vector<unsigned char> host_cipher_text = device_cipher_text;
    std::ofstream ofs("cipher_text.txt", std::ios::binary);

    ofs.write((char *)&host_cipher_text[0], numElements);

    ofs.close();

    return 0;
}
