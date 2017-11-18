//
// Created by Ayat ".ospanoff" Ospanov
//

#ifndef CONJUGATE_GRADIENT_METHOD
#define CONJUGATE_GRADIENT_METHOD

#include <vector>
#include <limits>

#include "CGMHelper.h"


////////////////////////////////////////////////////////////////////////////////
/// Problem prototype
////////////////////////////////////////////////////////////////////////////////
class Problem {
public:
    double A1, A2, B1, B2;
    double eps;
public:
    virtual double F(double x, double y) const = 0;
    virtual double phi(double x, double y) const = 0;
    virtual void computeR(const Func2D &p, Func2D &r, const Grid1D &gridX, const Grid1D &gridY) = 0;
    virtual void computeG(double alpha, const Func2D &r, Func2D &g) = 0;
    virtual double computeP(double tau, const Func2D &g, Func2D &p, const Grid1D &gridX, const Grid1D &gridY) = 0;
    virtual Fraction computeTau(const Func2D &r, const Func2D &g, const Grid1D &gridX, const Grid1D &gridY) = 0;
    virtual Fraction computeAlpha(const Func2D &r, const Func2D &g, const Grid1D &gridX, const Grid1D &gridY) = 0;
};

////////////////////////////////////////////////////////////////////////////////
/// ConjugateGradientMethod
////////////////////////////////////////////////////////////////////////////////
class ConjugateGradientMethod {
    Problem &problem;
    Func2D p, r, g;

    int numOfPointsX, numOfPointsY;
    MPIHelper helper;
    Grid1D gridX, gridY;

    std::vector<Exchanger> communication;

    int rankX;
    int rankY;
    int numOfProcX;
    int numOfProcY;

    int beginEdgeX;
    int endEdgeX;
    int beginEdgeY;
    int endEdgeY;

    double diff;

    int numIter;
    double timeElapsed;
public:
    ConjugateGradientMethod(Problem &problem, int N_x, int N_y, MPIHelper helper)
            :problem(problem),numOfPointsX(N_x),numOfPointsY(N_y),
             helper(helper),
             gridX(problem.A1, problem.A2, N_x),gridY(problem.B1, problem.B2, N_y) {
        init();
        diff = std::numeric_limits<double>::max();
        numIter = 0;
    }

    /// MPI set ups
    void init();

    void count_set_processes();
    void count_edges(int numOfPts, int numOfBlocks, int blockIdx, int &begin, int &end);
    void set_edges();

    bool has_left_neighbour() const { return rankX > 0; }
    bool has_right_neighbour() const { return rankX < numOfProcX - 1; }
    bool has_top_neighbour() const { return rankY > 0; }
    bool has_bottom_neighbour() const { return rankY < numOfProcY - 1; }
    int get_1D_rank(int x, int y) const { return y * numOfProcX + x; }

    void communicate(Func2D &data);

    void collectFraction(Fraction &f);
    void collectDouble(double &num);

    void collectP();
    void collectResults();

    /// The method
    void solve();
    void initialStep();
    void iteration();
};


#endif //CONJUGATE_GRADIENT_METHOD
