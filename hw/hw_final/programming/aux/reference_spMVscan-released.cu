#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <assert.h>
#include <string>

#include <cmath>
#include <ctime>

using std::cout;
using std::endl;
using std::ostream;
using std::string;

#include "mp1-util.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>

/*
 A sub-optimal serial solution
 to the sparse MV problem we provided
 */
void sparseMV_scan(matrix *m) {
	assert(m->a.size() == m->b.size());
	assert(m->a.size() == m->k.size());
	for (int k = 0; k < m->s.size()-1; k++) {
		const int j0 = m->s[k];
		assert(j0 < m->a.size());
		assert(m->k[j0] < m->x.size());
		assert(m->k[j0] >= 0);
		m->b[j0] = m->a[j0] * m->x[m->k[j0]];
		for (int j = m->s[k]+1; j < m->s[k+1]; j++) {
			assert(m->k[j] < m->x.size());
			assert(m->k[j] >= 0);		
			m->b[j] = m->b[j-1] + m->a[j] * m->x[m->k[j]];
		}    
	}
    return;
}

void output(std::vector<double> & a, const char *outFile) {
	std::ofstream out(outFile, std::ios::out);
	out.precision(16);
    for (int i=0;i<a.size()-1;++i)
    	out << std::scientific << a[i] << " ";
    out << a[a.size()-1] << endl;
}

	// Find the L2 norm of a vector
double L2norm(std::vector<double> & a) {
    if (a.size() <= 0) {
		fprintf(stderr, "The vector has size: %lu\n",a.size());
		exit(0);
    }
	
    double sum = 0.0;
    for(int i = 0; i < a.size(); i++) {
		sum += a[i]*a[i];
    }
	
    return std::sqrt(sum);
}

	// ||b-a||
double L2Distance(std::vector<double> & a, std::vector<double> & b) {
    if (a.size() != b.size()) {
		fprintf(stderr, "The vectors have different sizes: %lu %lu\n",
				a.size(), b.size());
		exit(0);
    }
	
    std::vector<double> c(a.size());
    for(int i = 0; i < a.size(); i++) c[i] = a[i]-b[i];
	
	return L2norm(c);
}

	// ||b-a|| / ||a||
double relativeL2Error(std::vector<double> & a, std::vector<double> & b) {
    double norm_a = L2norm(a);
	
    if (norm_a <= 0) {
		fprintf(stderr, "The L^2 norm of a is: %g\n",norm_a);
		exit(0);
    }
	
    return L2Distance(a,b)/norm_a;
}

	// Find the Linf norm of a vector
double LInfnorm(std::vector<double> & a) {
    if (a.size() <= 0) {
		fprintf(stderr, "The vector has size: %lu\n",a.size());
		exit(0);
    }
	
    double max = 0.0;
    for(int i = 0; i < a.size(); i++) {
		double ai = std::abs(a[i]);
		max = (ai > max ? ai : max);
    }
	
    return max;
}

	// ||b-a||
double LInfDistance(std::vector<double> & a, std::vector<double> & b) {
    if (a.size() != b.size()) {
		fprintf(stderr, "The vectors have different sizes: %lu %lu\n",
				a.size(), b.size());
		exit(0);
    }
	
    std::vector<double> c(a.size());
    for(int i = 0; i < a.size(); i++) c[i] = a[i]-b[i];
	
	return LInfnorm(c);
}

	// ||b-a|| / ||a||
double relativeLInfError(std::vector<double> & a, std::vector<double> & b) {
    double norm_a = LInfnorm(a);
	
    if (norm_a <= 0) {
		fprintf(stderr, "The L^Inf norm of a is: %g\n",norm_a);
		exit(0);
    }
    return LInfDistance(a,b)/norm_a;
}

template <class T> ostream &operator<<(ostream &out,
									   thrust::host_vector<T> & a) {
	for (int i=0;i<a.size()-1;++i)
    out << a[i] << " ";
	out << a[a.size()-1];
	return out;
}

template <class T> ostream &operator<<(ostream &out,
									   thrust::device_vector<T> & a) {
	for (int i=0;i<a.size()-1;++i)
    out << a[i] << " ";
	out << a[a.size()-1];
	return out;
}

int main(int argc, char **argv) {
		// Root directory
	string root = "/home/dbtsai/benchmarksuite";
	root += "/";
	
		// List of directory names to test
	std::vector<string> dir(15);
	
	dir[0]  = "cant";
	dir[1]  = "consph";
	dir[2]  = "cop20k_A";
	dir[3]  = "dense2";
	dir[4]  = "jonheart";
	dir[5]  = "mac_econ_fwd500";
	dir[6]  = "mc2depi";
	dir[7]  = "pdb1HYS";
	dir[8]  = "pwtk";
	dir[9]  = "qcd5_4";	
	dir[10] = "rail4284";
	dir[11] = "rma10";
	dir[12] = "scircuit";
	dir[13] = "shipsec1";
	dir[14] = "webbase-1M";
	//dir[15] = "../reference_solution";

	printf("\nRoot directory for data files: %s\n",root.c_str());		
	
		// Loop through directories checking all the calculations
	
	for (int dir_index = 0; dir_index < dir.size(); ++dir_index) {


		printf("\nReading data in directory: %s\n",dir[dir_index].c_str());		

		string dir_file = root + dir[dir_index];
		string a_txt = dir_file + "/a.txt";
		string x_txt = dir_file + "/x.txt";
		string b_txt = dir_file + "/b.txt";		

			// Read in from a.txt, x.txt, and b_txt
		matrix *m = load(a_txt.c_str(), x_txt.c_str(), b_txt.c_str());

		printf("Number of iterations set to: N = %d\n",m->N);

		std::vector<double> a0 = m->a;
		
			// b stores the solution read from the file
		std::vector<double> b_file = m->b;

			// Computing the solution using the CPU implementation
		clock_t workTimer = clock();
		for(int i = 0; i < m->N; i++) {
			// cout << i << " " << m->N << endl;
			sparseMV_scan(m);
			if(i != m->N - 1) m->b.swap(m->a);
		}
			// m->b contains the CPU solution.
		clock_t totalWorkTimer = clock() - workTimer;
		printf("CPU   took %g ms\n", 
			   1e3*totalWorkTimer / CLOCKS_PER_SEC);
		std::vector<double> b_cpu = m->b;
		
		/* Write back the CPU solution to b.txt.
		   This is in case b.txt does not contain the correct solution. */
	//	 output(b_cpu,b_txt.c_str());
		
		event_pair timer;
		
			// Calculate a slow GPU solution using thrust
		typedef thrust::device_vector<float> tvector;
		typedef thrust::device_vector<int> tivector;
		tvector d_a = a0;
		tvector d_b = m->a;
		tvector d_x = m->x;  
		tivector d_s = m->s;
		tivector d_k = m->k;
		
		// thrust code
		
		//start_timer(&timer);
		for(int i = 0; i < m->N; i++) {
			// more thrust code
			d_a = d_b;
		}  
		//stop_timer(&timer,"GPU 1");
		
			// Copy back
		thrust::host_vector<float> h_b = d_b;
		std::vector<double> b_gpu(h_b.size());
		for (int i=0; i<b_gpu.size(); ++i) b_gpu[i] = h_b[i];
		
			// Thrust implementation using segmented scans
			// Re-initialize
		d_a = a0;
		
		// Second thrust code

		//start_timer(&timer);
		for(int i = 0; i < m->N; i++) {
			// more thrust code
			d_a = d_b;
		}  
		//stop_timer(&timer,"GPU 2");  
		
			// Copy back
		h_b = d_b;
		std::vector<double> b_thrust(h_b.size());
		for (int i=0; i<b_gpu.size(); ++i) b_thrust[i] = h_b[i];
		
			// Printing all error estimates
		printf("File:\t absolute l2/linf - relative l2/linf \t%8.1e %8.1e %8.1e %8.1e\n",
			   L2Distance(b_cpu,b_file), LInfDistance(b_cpu,b_file),
			   relativeL2Error(b_cpu,b_file), relativeLInfError(b_cpu,b_file));  
		
	//	printf("GPU 1:\t absolute l2/linf - relative l2/linf \t%8.1e %8.1e %8.1e %8.1e\n",
	//		   L2Distance(b_cpu,b_gpu), LInfDistance(b_cpu,b_gpu),
	//		   relativeL2Error(b_cpu,b_gpu), relativeLInfError(b_cpu,b_gpu));   
		
	//	printf("GPU 2:\t absolute l2/linf - relative l2/linf \t%8.1e %8.1e %8.1e %8.1e\n",
	//		   L2Distance(b_cpu,b_thrust), LInfDistance(b_cpu,b_thrust),
	//		   relativeL2Error(b_cpu,b_thrust), relativeLInfError(b_cpu,b_thrust));   
		
	//	printf("GPU 1/2: absolute l2/linf - relative l2/linf \t%8.1e %8.1e %8.1e %8.1e\n",
	//		   L2Distance(b_gpu,b_thrust), LInfDistance(b_gpu,b_thrust),
	//		   relativeL2Error(b_gpu,b_thrust), relativeLInfError(b_gpu,b_thrust));  

		delete m;
	}
	return 0;
}
