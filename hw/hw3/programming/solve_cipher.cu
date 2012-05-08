/* CUDA Machine Problem 3 - Thrust
 *
 * Vignere Ciphers
 * Part 2 - Cracking
 * 
 * Your Assignment:
 * 
 * 1) Generate the frequency table for the cipher text
 * 2) Generate the bi-gram frequency table (ie aa, th, qu, st, ...)
 * 3) Determine the key length using the IOC
 * 4) Decode the cipher text and output the plain text
 * 5) You can only use thrust algorithms
 *
 */


#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <fstream>
#include <iostream>

//You will find this strided_range iterator useful
//foo = [0 1 2 3 4 5 6 7 8]
//strided_range(foo.begin(), 2) -> [0, 2, 4, 6, 8]
//strided_range(foo.begin() + 1, 2) -> [1, 3, 5, 7]
template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }
    
    protected:
    difference_type stride;
    Iterator first;
    Iterator last;
};

struct apply_shift : thrust::binary_function<unsigned char, int, unsigned char> {
    //TODO: fill in
};

int main(int argc, char **argv) {
    //First load the text 
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

    //we assume the cipher text has been sanitized
    //generate the frequency table
    //print out all 26 letters and their frequency
    //a: .03
    //b: .02
    //...

    //generate the digraph frequency table
    //print out the top 20
    //kh: .001
    //tl: .0009
    //...

    //now we need to crack vignere cipher
    //first we need to determine the key length
    //use the index of coincidence
    int keyLength = 0;
    {
        bool found = false;
        int i = 1;
        while (!found) {
            int numMatches; //TODO: set this to the correct value

            double ioc = numMatches / ((double)(text.size() - i) / 26.); 

            if (ioc > 1.6) {
                if (keyLength == 0)
                    keyLength = i;
                else if (2 * keyLength == i)
                    found = true;
                else {
                    std::cout << "Unusual pattern in text!" << std::endl;
                    exit(1);
                }
            }
            ++i; 
        }
    }

    std::cout << "keyLength: " << keyLength << std::endl;

    //once we know the key length, then we can do frequency analysis on each pos mod length
    //allowing us to easily break each cipher independently
    thrust::device_vector<int> dShifts(keyLength);
    typedef thrust::device_vector<unsigned char>::iterator Iterator;
    for (int i = 0; i < keyLength; ++i) {
        //TODO: set the dShifts vector correctly
    }

    thrust::device_vector<unsigned char> d_plain_text(text.size());

    //TODO: use the dShifts vector to generate the plaintext

    thrust::host_vector<unsigned char> h_plain_text = d_plain_text; 

    std::ofstream ofs("plain_text.txt", std::ios::binary);

    ofs.write((char *)&h_plain_text[0], h_plain_text.size());

    ofs.close();

    return 0;
}
