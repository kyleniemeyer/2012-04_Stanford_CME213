#include <iostream>
#include <fstream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/remove.h>
#include <thrust/random.h>

struct isnot_lowercase_alpha : thrust::unary_function<bool, unsigned char>{
    __host__ __device__
    bool operator()(const unsigned char &c) {
        return c < 97 || c > 122;
    }
};

struct upper_to_lower : thrust::unary_function<unsigned char, unsigned char>{
    __host__ __device__
    unsigned char operator()(const unsigned char &c) {
        if (c >= 65 && c <= 90)
            return c + 32;
        else
            return c;
    }
};

struct apply_shift : thrust::binary_function<unsigned char, int, unsigned char> {
    unsigned int period;
    unsigned int *shifts;

    __host__ __device__
    apply_shift(unsigned int p, unsigned int *s) : period(p), shifts(s) {}

    __host__ __device__
    unsigned char operator()(const unsigned char &c, int pos) {
        unsigned char new_c = c + shifts[pos % period];
        if (new_c > 122)
            new_c -= 26;
        else if (new_c < 97)
            new_c += 26;
        return new_c;
    }
};

int main(int argc, char **argv) {
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

    //sanitize input to contain only a-z lowercase
    thrust::device_vector<unsigned char> dText = text;
    thrust::device_vector<unsigned char> text_clean(text.size());

    int numElements = thrust::remove_copy_if(
                                             thrust::make_transform_iterator(dText.begin(), upper_to_lower()), 
                                             thrust::make_transform_iterator(dText.end(), upper_to_lower()), 
                                             text_clean.begin(), isnot_lowercase_alpha()) - text_clean.begin();

    unsigned int period = atoi(argv[2]);

    thrust::device_vector<unsigned int> shifts(period);
    thrust::default_random_engine rng(123);
    thrust::uniform_int_distribution<int> uniform_dist(1, 25);

    for (int i = 0; i < shifts.size(); ++i) {
        shifts[i] = uniform_dist(rng); //don't allow 0 shifts
    }

    thrust::device_vector<unsigned char> device_cipher_text(numElements);

    thrust::transform(text_clean.begin(), text_clean.begin() + numElements, thrust::make_counting_iterator((int)0),
                      device_cipher_text.begin(), apply_shift(period, thrust::raw_pointer_cast(&shifts[0])));

    thrust::host_vector<unsigned char> host_cipher_text = device_cipher_text;
    std::ofstream ofs("cipher_text.txt", std::ios::binary);

    ofs.write((char *)&host_cipher_text[0], numElements);

    ofs.close();

    return 0;
}
