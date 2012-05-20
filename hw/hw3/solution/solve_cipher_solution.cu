#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <fstream>
#include <iostream>

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

//from strided range example
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
    unsigned int period;
    int *shifts;

    __host__ __device__
    apply_shift(unsigned int p, int *s) : period(p), shifts(s) {}

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
    thrust::device_vector<unsigned char> text_clean = text;
    thrust::device_vector<unsigned char> text_sorted = text_clean;

    thrust::sort(text_sorted.begin(), text_sorted.end());

    thrust::device_vector<int> letterHisto(26);
    thrust::device_vector<unsigned char> letters(26);
    //here we assume all 26 letters appear in the text, which is probably a decent assumption
    thrust::reduce_by_key(text_sorted.begin(), text_sorted.end(), thrust::make_constant_iterator( (int)1),
                          letters.begin(), letterHisto.begin() );

    for (int i = 0; i < 26; ++i) {
        std::cout << letters[i] << " " << letterHisto[i] / (double)text_clean.size() << std::endl;
    }

    //we can cast to an unsigned short and then sort
    thrust::device_vector<unsigned char> text_digraph = text_clean;
    thrust::device_ptr<unsigned short> tcb((unsigned short *)thrust::raw_pointer_cast(&text_digraph[0]));
    int numElements = text_clean.size() / 2; //if it is odd and we leave out the last element, oh well
    
    thrust::sort(tcb, tcb + numElements);

    thrust::device_vector<int> digraphHisto(26 * 26);
    thrust::device_vector<unsigned short> digraphs(26 * 26);

    //here we cannot assume that all possible digraphs appear, since some are basically
    //non-existent even in the longest texts  zq, xq, qv, ...
    thrust::pair<thrust::device_vector<unsigned short>::iterator, thrust::device_vector<int>::iterator> outputIt = 
                thrust::reduce_by_key(tcb, tcb + numElements, thrust::make_constant_iterator( (int)1),
                                      digraphs.begin(), digraphHisto.begin());

    //sort the digraphs by their frequency and display the top 20
    thrust::sort_by_key(digraphHisto.begin(), outputIt.second, digraphs.begin(), thrust::greater<int>());

    for (int i = 0; i  < 20; ++i) {
        std::cout << (char)(digraphs[i] & 0x00FF) << (char)( (digraphs[i] & 0xFF00) >> 8) << " " << digraphHisto[i] / (double)numElements << std::endl;
    }

    //now we need to crack vignere cipher
    //first we need to determine the key length
    //use the kappa index of coincidence
    int keyLength = 0;
    {
        bool found = false;
        int i = 1;
        while (!found) {
            int numMatches = thrust::inner_product(text_clean.begin() + i, text_clean.end(), text_clean.begin(), 
                                                   0, thrust::plus<int>(), thrust::equal_to<unsigned char>()); 

            double ioc = numMatches / ((double)(text_clean.size() - 1) / 26.); 

            std::cout << "Ioc: " << ioc << std::endl;
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
    thrust::device_vector<unsigned char> text_copy = text_clean;
    thrust::device_vector<int> dShifts(keyLength);
    typedef thrust::device_vector<unsigned char>::iterator Iterator;
    for (int i = 0; i < keyLength; ++i) {
        strided_range<Iterator> it(text_copy.begin() + i, text_copy.end(), keyLength);
        thrust::sort(it.begin(), it.end());

        thrust::device_vector<int> letterHisto1(26);
        thrust::device_vector<unsigned char> letters1(26);
        //here we assume all 26 letters appear in the text, which is probably a decent assumption
        thrust::reduce_by_key(it.begin(), it.end(), thrust::make_constant_iterator( (int)1),
                              letters1.begin(), letterHisto1.begin() );

        int maxLoc = thrust::max_element(letterHisto1.begin(), letterHisto1.end()) - letterHisto1.begin();
        dShifts[i] = -(maxLoc - 4);
    }

    thrust::device_vector<unsigned char> d_plain_text(text_clean.size());

    thrust::transform(text_clean.begin(), text_clean.end(), thrust::make_counting_iterator((int)0), 
                      d_plain_text.begin(), apply_shift(keyLength, thrust::raw_pointer_cast(&dShifts[0])));

    thrust::host_vector<unsigned char> h_plain_text = d_plain_text; 

    std::ofstream ofs("plain_text.txt", std::ios::binary);

    ofs.write((char *)&h_plain_text[0], h_plain_text.size());

    ofs.close();

    return 0;
}
