 /* 2D Heat Diffusion w/MPI
 * 
 * You will implement the familiar 2D Heat Diffusion from the
 * previous homework on CPUs with MPI.
 * 
 * You have been given the simParams class updated
 * with all necessary parameters and the outline of
 * Grid class that you fill in.  You are also given the 
 * stencil calculations since you have already implemented
 * them in the previous homework.
 *
 * You are also given a macro - MPI_SAFE_CALL which you should
 * wrap all MPI calls with to always check error return codes.
 *
 * You will implement and investigate 2 different domain 
 * decompositions techniques: 1-D stripes and 2-D squares.
 * 
 * You will also investigate the impact of synchronous vs. asynchronous
 * communication.  To minimize programming effort and time you only need
 * to implement asynchronous communication and then implement synchronous
 * communication on top of the asynchronous routine by waiting for
 * the communication to finish.
 *
 * This means you will only use the asyncrhonous MPI communication routines
 * How would the communication pattern have to be different if you used the
 * synchronous communication routines for the synchronous communication.
 * 
 */

#include <ostream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <fstream>
#include <string>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdlib.h>

#include "mpi.h"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

class simParams {
    public:
        simParams(const char *filename, bool verbose); //parse command line
                                                       //does no error checking
        simParams(); //use some default values

        int    nx()         const {return nx_;}
        int    ny()         const {return ny_;}
        double lx()         const {return lx_;}
        double ly()         const {return ly_;}
        double alpha()      const {return alpha_;}
        int    iters()      const {return iters_;}
        double dx()         const {return dx_;}
        double dy()         const {return dy_;}
        double ic()         const {return ic_;}
        int    order()      const {return order_;}
        double xcfl()       const {return xcfl_;}
        double ycfl()       const {return ycfl_;}
        int    gridMethod() const {return gridMethod_;}
        bool   sync()       const {return synchronous_;}
        double topBC()      const {return bc[0];}
        double leftBC()     const {return bc[1];}
        double bottomBC()   const {return bc[2];}
        double rightBC()    const {return bc[3];}

    private:
        int    nx_, ny_;     //number of grid points in each dimension
        double lx_, ly_;     //extent of physical domain in each dimension
        double alpha_;       //thermal conductivity
        double dt_;          //timestep
        int    iters_;       //number of iterations to do
        double dx_, dy_;     //size of grid cell in each dimension
        double ic_;          //uniform initial condition
        double xcfl_, ycfl_; //cfl numbers in each dimension
        int    order_;       //order of discretization
        int    gridMethod_;  //1-D or 2-D
        bool   synchronous_; //Sync or Async communication scheme
        double bc[4];        //0 is top, counter-clockwise

        void calcDtCFL();
};

simParams::simParams() {
    nx_ = ny_ = 10;
    lx_ = ly_ = 1;
    alpha_ = 1;
    iters_ = 1000;
    order_ = 2;

    dx_ = lx_ / (nx_ - 1);
    dy_ = ly_ / (ny_ - 1);

    ic_ = 5.;

    gridMethod_ = 1;

    synchronous_ = true;

    bc[0] = 0.;
    bc[1] = 10.;
    bc[2] = 0.;
    bc[3] = 10.;

    calcDtCFL();
}

simParams::simParams(const char *filename, bool verbose) {
    std::ifstream ifs(filename);

    if (!ifs.good()) {
        std::cerr << "Couldn't open parameter file!" << std::endl;
        exit(1);
    }

    ifs >> nx_ >> ny_;
    ifs >> lx_ >> ly_;
    ifs >> alpha_;
    ifs >> iters_;
    ifs >> order_;
    ifs >> ic_;
    ifs >> gridMethod_;
    ifs >> synchronous_;
    ifs >> bc[0] >> bc[1] >> bc[2] >> bc[3];

    ifs.close();

    dx_ = lx_ / (nx_ - 1);
    dy_ = ly_ / (ny_ - 1);

    calcDtCFL();

    int rank;

    MPI_SAFE_CALL( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
    if (verbose && rank == 0) {
        printf("nx: %d ny: %d\nlx %f: ly: %f\nalpha: %f\niterations: %d\norder: %d\nic: %f\nsync: %d\n", 
                nx_, ny_, lx_, ly_, alpha_, iters_, order_, ic_, synchronous_);
        printf("domainDecomp: %d\ntopBC: %f lftBC: %f botBC: %f rgtBC: %f\ndx: %f dy: %f\ndt: %f xcfl: %f ycfl: %f\n", 
                gridMethod_, bc[0], bc[1], bc[2], bc[3], dx_, dy_, dt_, xcfl_, ycfl_);
    }
}

void simParams::calcDtCFL() {
    //check cfl number and make sure it is ok
    if (order_ == 2) {
        //make sure we come in just under the limit
        dt_ = (.5 - .0001) * (dx_ * dx_ * dy_ * dy_) / (alpha_ * (dx_ * dx_ + dy_ * dy_));
        xcfl_ = (alpha_ * dt_) / (dx_ * dx_);
        ycfl_ = (alpha_ * dt_) / (dy_ * dy_);
    }
    else if (order_ == 4) {
        dt_ = (.5 - .0001) * (12 * dx_ * dx_ * dy_ * dy_) / (16 * alpha_ * (dx_ * dx_ + dy_ * dy_));
        xcfl_ = (alpha_ * dt_) / (12 * dx_ * dx_);
        ycfl_ = (alpha_ * dt_) / (12 * dy_ * dy_);
    }
    else if (order_ == 8) {
        dt_ = (.5 - .0001) * (5040 * dx_ * dx_ * dy_ * dy_) / (8064 * alpha_ * (dx_ * dx_ + dy_ * dy_));
        xcfl_ = (alpha_ * dt_) / (5040 * dx_ * dx_);
        ycfl_ = (alpha_ * dt_) / (5040 * dy_ * dy_);
    }
    else {
        std::cerr << "Unsupported discretization order. \
                      TODO: implement exception" << std::endl;
    }
}

class Grid {
    public:
        Grid(const simParams &params, bool debug);
        ~Grid() { }

        typedef int gridState;

        enum MessageTag {
            TOP_TAG,
            BOT_TAG,
            LFT_TAG,
            RGT_TAG
        };

        int gx() const {return gx_;}
        int gy() const {return gy_;}
        int nx() const {return nx_;}
        int ny() const {return ny_;}
        int borderSize() const {return borderSize_;}
        int rank() const {return ourRank_;}
        const gridState & curr() const {return curr_;}
        const gridState & prev() const {return prev_;}
        void swapState() {prev_ = curr_; curr_ = (curr_ + 1) & 1;} 

        //for speed doesn't do bounds checking
        double operator()(const gridState & selector, 
                                 int xpos, int ypos) const {
            return grid_[selector * gx_ * gy_ + ypos * gx_ + xpos];
        }

        double& operator()(const gridState & selector, 
                                  int xpos, int ypos) {
            return grid_[selector * gx_ * gy_ + ypos * gx_ + xpos];
        }

        void transferHaloDataASync();
        void waitForSends(); //block until sends are finished
        void waitForRecvs(); //block until receives are finished

        void saveStateToFile(std::string identifier) const;

        friend std::ostream & operator<<(std::ostream &os, const Grid& grid);

    private:
        std::vector<double> grid_;
        int gx_, gy_;             //total grid extents - non-boundary size + halos
        int nx_, ny_;             //non-boundary region
        int borderSize_;          //number of halo cells

        int procLeft_;            //MPI processor numbers
        int procRight_;           //of our neighbors
        int procTop_;             //negative if not used
        int procBot_;

        gridState curr_;
        gridState prev_;

        int ourRank_;
        bool debug_;

        std::vector<MPI_Request> send_requests_;
        std::vector<MPI_Request> recv_requests_;

        //prevent copying and assignment since they are not implemented
        //and don't make sense for this class
        Grid(const Grid &);
        Grid& operator=(const Grid &);

};

std::ostream& operator<<(std::ostream& os, const Grid &grid) {
    os << std::setprecision(3);
    for (int y = grid.gy() - 1; y != -1; --y) {
        for (int x = 0; x < grid.gx(); x++) {
            os << std::setw(5) << grid(grid.curr(), x, y) << " ";
        }
        os << std::endl;
    }
    os << std::endl;
    return os;
}

Grid::Grid(const simParams &params, bool debug) {
    debug_ = debug;

    curr_ = 1;
    prev_ = 0;

    //need to figure out which processor we are and who our neighbors are...
    int totalNumProcessors;
    //TODO: set ourRank_ and totalNumProcessors

    //based on total number of processors and grid configuration
    //determine our neighbors
    procLeft_ = -1;
    procRight_ = -1;
    procTop_ = -1;
    procBot_ = -1;

    //1D decomposition - horizontal stripes
    if (params.gridMethod() == 1) {
        //TODO: set proc* and nx, ny correctly
    }
    else if (params.gridMethod() == 2) { //2D decomposition
                                         //you are only required to implement
                                         //decomposition for square grids of processors
                                         //ie 1x1, 2x2, 3x3, etc.
                                         //handling of arbitrary # of processors
                                         //is extra credit
        //TODO: set proc* and nx, ny correctly
    }
    else {
        std::cerr << "Unsupported grid decomposition method! " << params.gridMethod() << std::endl;
        exit(1);
    }
    
    if (params.order() == 2) 
        borderSize_ = 1;
    else if (params.order() == 4)
        borderSize_ = 2;
    else if (params.order() == 8)
        borderSize_ = 4;

    assert(nx_ > 2 * borderSize_);
    assert(ny_ > 2 * borderSize_);

    //TODO: set gx and gy correctly
   
    if (debug) { 
        printf("%d: (%d, %d) (%d, %d) lft: %d rgt: %d top: %d bot: %d\n", \
                ourRank_, nx_, ny_, gx_, gy_, procLeft_, procRight_, procTop_, procBot_);
    }

    //resize and set ICs
    grid_.resize(gx_ * gy_, params.ic());

    //set BCs
    //TODO: fill in locations in grid_ with the correct boundary conditions

    //create the copy of the grid we need for ping-ponging
    grid_.insert(grid_.end(), grid_.begin(), grid_.end());

}

void Grid::waitForSends() {
    //TODO
}

void Grid::waitForRecvs() {
    //TODO
}

//sends from previous to current
void Grid::transferHaloDataASync() {
    //we send from the prev grid and receive into the current grid
    //TODO
}

void Grid::saveStateToFile(std::string identifier) const {
    std::stringstream ss;
    ss << "grid" << ourRank_ << "_" << identifier << ".txt";
    std::ofstream ofs(ss.str().c_str());
    
    ofs << *this << std::endl;

    ofs.close();
}

inline double stencil2(const Grid &grid, int x, int y, double xcfl, double ycfl, const Grid::gridState &prev) {
    return grid(prev, x, y) + 
           xcfl * (grid(prev, x+1, y) + grid(prev, x-1, y) - 2 * grid(prev, x, y)) + 
           ycfl * (grid(prev, x, y+1) + grid(prev, x, y-1) - 2 * grid(prev, x, y));
}

inline double stencil4(const Grid &grid, int x, int y, double xcfl, double ycfl, const Grid::gridState &prev) {
    return grid(prev, x, y) + 
           xcfl * (   -grid(prev, x+2, y) + 16 * grid(prev, x+1, y) -
                    30 * grid(prev, x, y) + 16 * grid(prev, x-1, y) - grid(prev, x-2, y)) + 
           ycfl * (   -grid(prev, x, y+2) + 16 * grid(prev, x, y+1) -
                    30 * grid(prev, x, y) + 16 * grid(prev, x, y-1) - grid(prev, x, y-2));
}

inline double stencil8(const Grid &grid, int x, int y, double xcfl, double ycfl, const Grid::gridState &prev) {
    return grid(prev, x, y) +
           xcfl*(-9*grid(prev,x+4,y) + 128*grid(prev,x+3,y) - 1008*grid(prev,x+2,y) + 8064*grid(prev,x+1,y) -
                                                  14350*grid(prev, x, y) + 
                 8064*grid(prev,x-1,y) - 1008*grid(prev,x-2,y) + 128*grid(prev,x-3,y) - 9*grid(prev,x-4,y)) + 
           ycfl*(-9*grid(prev,x,y+4) + 128*grid(prev,x,y+3) - 1008*grid(prev,x,y+2) + 8064*grid(prev,x,y+1) -
                                                  14350*grid(prev,x,y) +
                8064*grid(prev,x,y-1) -1008*grid(prev,x,y-2) + 128*grid(prev,x,y-3) - 9*grid(prev,x,y-4));
}

void syncComputation(Grid &grid, const simParams &params) {
    //TODO
}

void asyncComputation(Grid &grid, const simParams &params) {
    //TODO
    //The whole point of the asynchronous communication to is do communication while
    //the border regions are being transferred.  You should structure this routine so that
    //the transfer starts, the computation on the inner region is performed, then the computation
    //on the halo region is performed after making sure the communication is finished.
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Please supply a parameter file!" << std::endl;
        exit(1);
    }

    MPI_Init(&argc, &argv);

    simParams params(argv[1], true);
    Grid grid(params, true);

    grid.saveStateToFile("init"); //save our initial state, useful for making sure we
                                  //got setup and BCs right

    double start = MPI_Wtime();

    if (params.sync()) {
        syncComputation(grid, params);
    }
    else {
        asyncComputation(grid, params);
    }

    double end = MPI_Wtime();

    if (grid.rank() == 0) {
        std::cout << params.iters() << " iterations on a " << params.nx() << " by " 
                  << params.ny() << " grid took: " << end - start << " seconds." << std::endl;
    }
    grid.saveStateToFile("final"); //final output for correctness checking of computation

    MPI_Finalize(); 
    return 0;
}
