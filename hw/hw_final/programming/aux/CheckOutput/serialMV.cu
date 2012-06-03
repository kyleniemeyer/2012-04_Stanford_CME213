#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>

#include <cmath>
#include <ctime>
#include "mp1-util.h"

// Thrust includes go here

/*
  @author Austin Gibbons : gibbons4@stanford.edu
  @date 5/30/12

  Post to CME 213 piazza with any doubts
*/

/*
  A sub-optimal serial solution
  to the sparse MV problem we provided
*/
void sparseMV_scan(matrix *m) {
  int index = 0;
  for (int j = 0; j < m->b.size(); j++) {
    m->b[j] = 0;
    if (j >= m->s[index+1]) index++;
    for (int i = m->s[index]; i <= j; i++) {
      m->b[j] += m->a[i] * m->x[m->k[i]];
    }
  }
  return ;
}

void output(std::vector<double> arr, char *outFile) {
  FILE * out = fopen(outFile, "wb");
  for(std::vector<double>::iterator it = arr.begin(); it < arr.end(); it++)
    fprintf(out, "%f ", *it);
  fprintf(out, "\n");

  return ;
}

// Find the euclidean distance
void L2norm(std::vector<double> a, std::vector<double> b) {
  if (a.size() != b.size()) {
    fprintf(stderr, "Finding the norm between vectors with different sizes, exiting\n");
    exit(0);
  }

  double sum = 0.0;
  for(int i = 0; i < a.size(); i++) {
    sum += (a[i]-b[i])*(a[i]-b[i]);
  }

  printf("L2-norm: %f\n", std::sqrt(sum));
}

// Find the max difference
void LInfnorm(std::vector<double> a, std::vector<double> b) {
  if (a.size() != b.size()) {
    fprintf(stderr, "Finding the norm between vectors with different sizes, exiting\n");
    exit(0);
  }

  double max = -1;
  for(int i = 0; i < a.size(); i++) {
    if(std::abs(a[i]-b[i]) > max) {
      max = std::abs(a[i]-b[i]);
    }
  }

  printf("Infinity Norm: %f\n", max);
}

// ||b-a|| / ||a||
void relativeError(std::vector<double> a, std::vector<double> b) {
  if (a.size() != b.size()) {
    fprintf(stderr, "Finding the norm between vectors with different sizes, exiting\n");
    exit(0);
  }

  double sum = 0.0;
  double diff = 0.0;
  for(int i = 0; i < a.size(); i++) {
    diff += (a[i]-b[i])*(a[i]-b[i]);
    sum += a[i]*a[i];
  }
  
  if (sum == 0.0) {
    fprintf(stderr, "Trying to divide by a zero vector, exiting\n");
    exit(0);
  }

  printf("Relative Error: %f\n", std::sqrt(diff) / std::sqrt(sum));
}

int main(int argc, char **argv) {
  if(argc != 4) { 
    fprintf(stderr, "nvcc serialMV.cu\n");
    fprintf(stderr, "./a.out a.txt x.txt b.txt\n");
    fprintf(stderr, "exiting\n");
    exit(0);
  }

  //read in from a.txt and x.txt
  matrix *m = load(argv[1], argv[2], argv[3]);

  std::vector<double> b = m->b;
  //event_pair timer;
  //start_timer(&timer);
  //serial
    clock_t workTimer = clock();
  for(int i = 0; i < m->N; i++) {
    printf("iteration: %d / %d\n", i+1, m->N);
    sparseMV_scan(m);
    if(i != m->N - 1) m->b.swap(m->a);
  }
  //stop_timer(&timer,"serial: ");
      clock_t totalWorkTimer = clock() - workTimer;
      fprintf(stderr, "Time: cycles: %ld\tseconds: %ld\n", totalWorkTimer, totalWorkTimer / CLOCKS_PER_SEC);
  // Thrust call(s) #1
  // Calculate the keys (for scanning)

  // Thrust call(s) #2
  // Generate a permutation iterator

  //start_timer(&timer);
  for(int i = 0; i < m->N; i++) {

    //Thrust call(s) #3
    //Create a transformation to multiply a and perm(x)

    //Thrust call(s) #4
    //perform the scan
  }
  //stop_timer(&timer,"parallel: ");
  //output(m->b, "serial.txt");

  //suppose the parallel solution is in std::vector<double> b
  L2norm(b, m->b);
  LInfnorm(b, m->b);

  //order matters for this one
  relativeError(m->b, b);
  return 0;
}
