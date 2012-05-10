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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
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

struct apply_shift : thrust::binary_function<unsigned char, unsigned int, unsigned char> {
    //TODO: fill in the functional
    __host__ __device__
    unsigned char operator()(const unsigned char &c, const unsigned int &shift)
    {
        return ((c - 'a') + (26-shift)) % 26 + 'a';
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
    
    thrust::device_vector<unsigned char> device_cipher_text = text;
    
    //we assume the cipher text has been sanitized
    //generate the frequency table
    //print out all 26 letters and their frequency
    //a: .03
    //b: .02
    //...

    // Since the number of bins is extreamly small (well, only 26) 
    // compared to the input size, using the binary search for dense 
    // histogram will be faster than the example using the reduced_by_key
    // in the lecture note. Here, I implemented the binary search version.
    
    thrust::device_vector<unsigned char> data = device_cipher_text;
    thrust::sort(data.begin(), data.end());
    thrust::device_vector<size_t> histogram(26);
    thrust::counting_iterator<unsigned char> counting_iter('a');
    thrust::upper_bound(data.begin(), data.end(), counting_iter, counting_iter + 26, histogram.begin());
    thrust::host_vector<size_t> host_histogram = histogram;
    std::cout<<"Text length: "<<length<<"\n\n";
    size_t result = 0;
    for(int i=0; i<26;i++)
    {  
        size_t count = 0;
        if( i == 0)
        { count = host_histogram[i]; }
        else
        { count = host_histogram[i]-host_histogram[i-1]; }
        result += count;
        std::cout <<(unsigned char) (i +'a')<<": "<<double(count)/double(length)<<"\n";
    }
    std::cout<<"\nSum of histogram: "<<double(result)/double(length) <<"\n\n";

    //generate the digraph frequency table
    //print out the top 20
    //kh: .001
    //tl: .0009
    //...

    thrust::host_vector<size_t>  host_data = device_cipher_text;
    thrust::host_vector<unsigned int> host_key(26*26);
    host_histogram.resize(26*26);
    for(int i=0; i<26*26; ++i) {host_histogram[i] = 0;host_key[i]=i;}
    // It seems that the most efficent way to do digraph table is 
    // doing the forloop. 
    for(size_t i=0; i<length-1; ++i)
    {
        unsigned char m = host_data[i]-'a';
        unsigned char n = host_data[i+1]-'a';
        host_histogram[m*26 + n]++;
    }
    thrust::sort_by_key(host_histogram.begin(), host_histogram.end(), host_key.begin(),thrust::greater<int>());
    size_t sum = thrust::reduce(host_histogram.begin(), host_histogram.end(), (size_t) 0, thrust::plus<int>());

    for(int i=0; i<20; i++)
    {
        unsigned char n = (host_key[i]%26) + 'a';
        unsigned char m = (host_key[i] - n + 'a')/26 + 'a';
        std::cout<<m<<n<<": "<<" "<<double(host_histogram[i])/double(sum)<<"\n";
    }
    
    //now we need to crack vignere cipher
    //first we need to determine the key length
    //use the index of coincidence
    unsigned int keyLength = 0;
    {
        bool found = false;
        int i = 1;
        while (!found) {
            //TODO: set this to the correct value
            int numMatches = thrust::inner_product( device_cipher_text.begin(), device_cipher_text.end()-i, device_cipher_text.begin()+i, 
                                                    (int)0,thrust::plus<int>(), thrust::equal_to<int>());
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

    std::cout << "\nkeyLength: " << keyLength << std::endl;

    //once we know the key length, then we can do frequency analysis on each pos mod length
    //allowing us to easily break each cipher independently
    thrust::device_vector<unsigned int> dShifts(keyLength);
    thrust::host_vector<unsigned char>   host_partial_data((device_cipher_text.size()+keyLength-1)/keyLength );
    thrust::device_vector<unsigned char> partial_data     ((device_cipher_text.size()+keyLength-1)/keyLength );
    ////TODO: set the dShifts vector correctly
    // typedef thrust::device_vector<unsigned char>::iterator Iterator;
    for (int i = 0; i < keyLength; ++i) 
    {        
        int edge_case_shift = 0;
        for(int j = 0; j< (device_cipher_text.size()+keyLength-1)/keyLength; ++j)
        {
            if( (j*keyLength+i) < device_cipher_text.size()){  host_partial_data[j] = text[j*keyLength + i];}
            else{  host_partial_data[j] = 0;edge_case_shift=1;}
        }
        partial_data = host_partial_data;
        
        thrust::sort(partial_data.begin(), partial_data.end()-edge_case_shift);
        thrust::device_vector<size_t> histogram(26);
        thrust::counting_iterator<unsigned char> counting_iter('a');
        thrust::upper_bound(partial_data.begin(), partial_data.end()-edge_case_shift, counting_iter, counting_iter + 26, histogram.begin());
        thrust::host_vector<size_t> host_histogram = histogram;
        thrust::host_vector<size_t> host_count(26);
        thrust::host_vector<size_t> host_key(26);
        for(int k=0; k<26;++k)
        {
            host_key[k]=k;
            size_t count = 0;
            if( k == 0){ count = host_histogram[k]; }
            else{ count = host_histogram[k]-host_histogram[k-1]; }    
            host_count[k] = count;
        }
        thrust::sort_by_key(host_count.begin(), host_count.end(), host_key.begin(),thrust::greater<int>());
        int shift = host_key[0] + 'a' - 'e'; 
        if(shift > 0){ dShifts[i] = shift;}
        else{ dShifts[i] = (shift + 26) ;}
    }
    std::cout<<"\nKey: ";
    for(int i=0;i<keyLength;++i){ std::cout<<(unsigned char) ((dShifts[i]==26)?'z': dShifts[i] +'a');}
    std::cout<<"\n\m";
        
    thrust::device_vector<unsigned char> d_plain_text(text.size());

    //TODO: use the dShifts vector to generate the plaintext
    thrust::transform_iterator<periodic_shifts_fun, thrust::counting_iterator<size_t>  > 
        periodic_shifts_iter = thrust::make_transform_iterator(  thrust::make_counting_iterator((size_t)0),
                                                                 periodic_shifts_fun(keyLength, thrust::raw_pointer_cast(&dShifts[0]))); 
                                                             
    thrust::transform(  device_cipher_text.begin(), 
                        device_cipher_text.end(), 
                        periodic_shifts_iter,
                        d_plain_text.begin(), apply_shift());

    thrust::host_vector<unsigned char> h_plain_text = d_plain_text; 

    std::ofstream ofs("plain_text.txt", std::ios::binary);
    
    ofs.write((char *)&h_plain_text[0], h_plain_text.size());

    ofs.close();

    return 0;
}
