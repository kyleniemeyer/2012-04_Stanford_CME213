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

struct isnot_lowercase_alpha : thrust::unary_function<bool, unsigned char>{
    //TODO: fill in this functional
};

struct upper_to_lower : thrust::unary_function<unsigned char, unsigned char>{
    //TODO: fill in this functional
};

//This functional has to be initialized with the period and (a pointer) to the table of
//shifts.  You will need a constructor.
struct apply_shift : thrust::binary_function<unsigned char, int, unsigned char> {
    //TODO: fill in the functional
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
    int numElements; //the number of characters in the cleaned output

    thrust::device_vector<unsigned int> shifts(period);

    //TODO: Use thrust's random number generation capability to initialize the shift vector

    thrust::device_vector<unsigned char> device_cipher_text(numElements);

    //TODO: Again, with one thrust call, create the cipher text from the plaintext

    thrust::host_vector<unsigned char> host_cipher_text = device_cipher_text;
    std::ofstream ofs("cipher_text.txt", std::ios::binary);

    ofs.write((char *)&host_cipher_text[0], numElements);

    ofs.close();

    return 0;
}
