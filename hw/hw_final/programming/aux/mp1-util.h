#include <memory>
#include <new>
#include <assert.h>

typedef struct matrix {
  int n, p, q, N;
  std::vector<int> s, k;
  std::vector<double> a, x, b;
} matrix ;

struct event_pair
{
  cudaEvent_t start;
  cudaEvent_t end;
};

inline void check_launch(char * kernel_name)
{
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    printf("error on %s kernel\n",kernel_name);
    printf("error was: %s\n", cudaGetErrorString(err));
    exit(1);
  }
}


inline void start_timer(event_pair * p)
{
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}


inline void stop_timer(event_pair * p, char * kernel_name)
{
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);
  
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  printf("%s took %.2f ms\n",kernel_name, elapsed_time);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
}

//the volatiles are needed to get correct output
//when compiling with optimization
//gcc seems to do something silly otherwise
bool AlmostEqual2sComplement(volatile double A, volatile double B, int maxUlps)
{
    // Make sure maxUlps is non-negative and small enough that the
    // default NAN won't compare as equal to anything.
    // assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);
    volatile int aInt = *(volatile int*)&A;
    // Make aInt lexicographically ordered as a twos-complement int
    if (aInt < 0)
        aInt = 0x80000000 - aInt;
    // Make bInt lexicographically ordered as a twos-complement int
    volatile int bInt = *(volatile int*)&B;
    if (bInt < 0)
        bInt = 0x80000000 - bInt;
    int intDiff = abs(aInt - bInt);
    if (intDiff <= maxUlps)
        return true;
    return false;
}

template <class T> ostream &operator<<(ostream &out,
                     std::vector<T> & a) {
  for (int i=0;i<a.size()-1;++i)
    out << a[i] << " ";
  out << a[a.size()-1];
  return out;
}

/* Loading all data from file */
matrix *load(const char *a_file, const char *x_file, const char *b_file) {
  matrix * m = new matrix;

  std::ifstream afs(a_file, std::ios::in);
  if (!afs.good()) {
    printf("We could not open file %s\n",a_file);
    exit(0);
  }
  std::ifstream xfs(x_file, std::ios::in);
  if (!xfs.good()) {
    printf("We could not open file %s\n",x_file);
    exit(0);    
  }  
  std::ifstream bfs(b_file, std::ios::in);
  if (!bfs.good()) {
    printf("We could not open file %s\n",b_file);
  }  

  afs >> m->n;
  afs >> m->p;
  afs >> m->q;
  afs >> m->N;

  assert(m->n>0);
  assert(m->p>0);  
  assert(m->q>0);  
  assert(m->N>0);

  printf("Parameters read from file:\nn (size of a) = %d, p (number of segments) = %d, ",
    m->n,m->p);
  printf("q (size of x) = %d\n",m->q);

  // Allocating memory
  m->a.resize(m->n);
  m->b.resize(m->n);
  m->k.resize(m->n);
  m->s.resize(m->p);
  m->x.resize(m->q);

  // Reading a
  for (int i=0; i<m->a.size(); ++i) {
    afs >> m->a[i];
  }

  // cout << m->a << endl;

  // Reading s
  for (int i=0; i<m->s.size(); ++i) {
    afs >> m->s[i];
    if (i>0) {
      if (m->s[i] <= m->s[i-1]) {
        printf("Error in input file %s\n",a_file);
        printf("Index i, s[i-1], s[i] = %d %d %d\n",i,m->s[i-1],m->s[i]);
      }
      assert(m->s[i] > m->s[i-1]);
    }
  }
  assert(m->s[0] == 0);
  assert(m->s[m->s.size()-1] == m->n);

  // cout << m->s << endl;  

  // Reading k
  for (int i=0; i<m->k.size(); ++i) {
    afs >> m->k[i];
    assert(m->k[i] < m->q);
    assert(m->k[i] >= 0);
  }

  // cout << m->k << endl;

  // Reading x
  for (int i=0; i<m->x.size(); ++i) {
    xfs >> m->x[i];
  }

  // cout << m->x << endl;

  // Reading b
  if (bfs.good()) {
    for (int i=0; i<m->b.size(); ++i) {
      bfs >> m->b[i];
    }
  }

  // cout << m->b << endl;

  return m;
}
