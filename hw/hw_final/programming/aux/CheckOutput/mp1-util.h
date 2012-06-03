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

/*
  An inelegant loading script
  You do not need to time this code.
*/
matrix *load(char *a_file, char *x_file, char *b_file) {
  matrix *m = (matrix *) malloc(sizeof(matrix));

  std::ifstream afs(a_file, std::ios::in);
  std::ifstream xfs(x_file, std::ios::in);
  std::ifstream bfs(b_file, std::ios::in);

  std::string line;
  //get the sizes in a hardcoded fashion
  getline(afs, line);

  int end = line.find(' ');
  int start = 0;
  m->n = atoi(line.substr(start, end).c_str());
  start = end+1;
  end = line.find(' ', start);
  m->p = atoi(line.substr(start, end).c_str());
  start = end+1;
  end = line.find(' ', start);
  m->q = atoi(line.substr(start, end).c_str()); 
  start = end+1;
  end = line.find(' ', start);
  m->N = atoi(line.substr(start, end).c_str());
  
  int val;
  double temp;

  std::getline(afs, line);
  std::istringstream iss(line);
  while(iss >> temp) {
    (m->a).push_back(temp);
  //  (m->b).push_back(0);
  }

  std::getline(afs, line);
  std::istringstream s_iss(line);
  while(s_iss >> val)
    (m->s).push_back(val);

  std::getline(afs, line);
  std::istringstream k_iss(line);
  while(k_iss >> val)
    (m->k).push_back(val);

  std::getline(xfs, line);
  std::istringstream x_iss(line);
  while(x_iss >> temp)
    (m->x).push_back(temp);

  std::getline(bfs, line);
  std::istringstream b_iss(line);
  while(b_iss >> temp)
    (m->b).push_back(temp);

  return m;
}
