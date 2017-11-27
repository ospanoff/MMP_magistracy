//
// Created by Ayat ".ospanoff" Ospanov
//

#include <cmath>
#include <fstream>
#include <cstdlib>

#include "ConjugateGradientMethod.h"
#include "MPIHelper.h"


double ConjugateGradientMethod::tau(const Func2D &f, const Func2D &g) const {
    return (f * g) / (~g * g);
}

double ConjugateGradientMethod::alpha(const Func2D &f, const Func2D &g) const {
    return (~f * g) / (~g * g);
}

ConjugateGradientMethod::ConjugateGradientMethod(Grid grid, mathFunction f, mathFunction phi, bool display,
                                                 mathFunction answer, int maxIters, double eps)
        :p(grid),r(grid),g(grid),
         F(grid, f),diff(grid, infty),
         eps(eps),display(display),trueAnswer(answer),
         maxIters(maxIters),numIters(1),solutionError(0)
{
    MPIHelper &helper = MPIHelper::getInstance();

    /// Compute p0
    if (!helper.hasLeftNeighbour()) {
        for (unsigned int j = 0; j < p.sizeY(); ++j) {
            p(0, j) = phi(grid.x[0], grid.y[j]);
        }
    } else {
        communicatingEdges.push_back(
                Exchanger(helper.getRank(helper.getRankX() - 1, helper.getRankY()),
                          EdgeIndexer(EdgeIndexer::ByX, 1, grid.y.size()),
                          EdgeIndexer(EdgeIndexer::ByX, 0, grid.y.size())
                )
        );
    }
    if (!helper.hasRightNeighbour()) {
        unsigned int right = p.sizeX() - 1;
        for (unsigned int j = 0; j < p.sizeY(); ++j) {
            p(right, j) = phi(grid.x[right], grid.y[j]);
        }
    } else {
        communicatingEdges.push_back(
                Exchanger(helper.getRank(helper.getRankX() + 1, helper.getRankY()),
                          EdgeIndexer(EdgeIndexer::ByX, grid.x.size() - 2, grid.y.size()),
                          EdgeIndexer(EdgeIndexer::ByX, grid.x.size() - 1, grid.y.size())
                )
        );
    }
    if (!helper.hasTopNeighbour()) {
        for (unsigned int i = 0; i < p.sizeX(); ++i) {
            p(i, 0) = phi(grid.x[i], grid.y[0]);
        }
    } else {
        communicatingEdges.push_back(
                Exchanger(helper.getRank(helper.getRankX(), helper.getRankY() - 1),
                          EdgeIndexer(EdgeIndexer::ByY, 1, grid.x.size()),
                          EdgeIndexer(EdgeIndexer::ByY, 0, grid.x.size())
                )
        );
    }
    if (!helper.hasBottomNeighbour()) {
        unsigned int bottom = p.sizeY() - 1;
        for (unsigned int i = 0; i < p.sizeX(); ++i) {
            p(i, bottom) = phi(grid.x[i], grid.y[bottom]);
        }
    } else {
        communicatingEdges.push_back(
                Exchanger(helper.getRank(helper.getRankX(), helper.getRankY() + 1),
                          EdgeIndexer(EdgeIndexer::ByY, grid.y.size() - 2, grid.x.size()),
                          EdgeIndexer(EdgeIndexer::ByY, grid.y.size() - 1, grid.x.size())
                )
        );
    }
}

void ConjugateGradientMethod::initialStep() {
    /// Compute r0
    r = ~p - F;
    r.synchronize(communicatingEdges);

    /// Compute g0
    g = r;

    /// Compute p1
    diff = r * tau(r, g);
    p -= diff;
    diffNorm = std::sqrt(diff * diff);
}

void ConjugateGradientMethod::iteration() {
    /// k = 1, 2, ...
    p.synchronize(communicatingEdges);

    /// r_{k} ~ p_{k}
    r = ~p - F;
    r.synchronize(communicatingEdges);

    /// alpha_{k} ~ r_{k}, g_{k-1}
    /// g_{k} ~ alpha_{k}
    g = r - g * alpha(r, g);
    g.synchronize(communicatingEdges);

    /// tau_{k+1} ~ r_{k}, g_{k}
    /// p_{k+1} = p_{k} - tau_{k+1} * g_{k}, k = 1, 2, ...
    diff = g * tau(r, g);
    p -= diff;
    diffNorm = std::sqrt(diff * diff);
}

void ConjugateGradientMethod::solve() {
    MPIHelper &helper = MPIHelper::getInstance();
    double start = helper.time();

    initialStep();
    while (diffNorm > eps && numIters < maxIters) {
        iteration();
        ++numIters;
        if (display) {
            double r_norm = std::sqrt(r * r);
            if (helper.isMaster()) {
                printf("#%d:\t ||residual|| = %lf;\t diff = %lf\n", numIters, r_norm, diffNorm);
            }
        }
    }

    timeElapsed = helper.time() - start;
}

void ConjugateGradientMethod::collectP() {
    MPIHelper &helper = MPIHelper::getInstance();

    p.synchronize(communicatingEdges);

    std::vector<double> recv;
    std::vector<int> recvSizes;
    std::vector<int> recvSizesX;
    std::vector<int> recvSizesY;
    std::vector<int> recvDispl;

    if (helper.isMaster()) {
        recvSizesX.resize(static_cast<unsigned int>(helper.getNumOfProcs()));
        recvSizesY.resize(static_cast<unsigned int>(helper.getNumOfProcs()));
        recvDispl.resize(static_cast<unsigned int>(helper.getNumOfProcs()), 0);
    }

    helper.GatherInts(p.sizeX(), recvSizesX);
    helper.GatherInts(p.sizeY(), recvSizesY);

    if (helper.isMaster()) {
        recvSizes.push_back(recvSizesX[0] * recvSizesY[0]);
        for (int i = 1; i < recvSizesX.size(); ++i) {
            recvSizes.push_back(recvSizesX[i] * recvSizesY[i]);
            recvDispl[i] = recvDispl[i - 1] + recvSizes[i - 1];
        }
        int k = static_cast<int>(recvDispl.size() - 1);
        recv.resize(static_cast<unsigned int>(recvDispl[k] + recvSizes[k]));
    }

    std::vector<double> f;
    for (unsigned int i = 0; i < p.sizeX(); ++i) {
        for (unsigned int j = 0; j < p.sizeY(); ++j) {
            f.push_back(p(i, j));
        }
    }
    helper.Gatherv(f, recv, recvSizes, recvDispl);

    if (helper.isMaster()) {
        Grid grid(p.grid.borders, Rect<unsigned int>(0, p.grid.numOfPointsX, 0, p.grid.numOfPointsY),
                  p.grid.numOfPointsX, p.grid.numOfPointsY);
        Func2D p_res(grid);

        int shiftX = 0;
        for (int rX = 0; rX < helper.getNumOfProcsX(); ++rX) {
            int shiftY = 0;
            for (int rY = 0; rY < helper.getNumOfProcsY(); ++rY) {
                int rank = helper.getRank(rX, rY);
                std::vector<double>::const_iterator it = recv.begin() + recvDispl[rank];
                for (int x = 0; x < recvSizesX[rank]; ++x) {
                    for (int y = 0; y < recvSizesY[rank]; ++y) {
                        int pX = shiftX + x;
                        int pY = shiftY + y;
                        if (
                                (x != 0 && y != 0 && x != recvSizesX[rank] - 1 && y != recvSizesY[rank] - 1) || /// Don't write all edges
                                (pX == 0 || pX == grid.numOfPointsX - 1 || pY == 0 || pY == grid.numOfPointsY - 1) /// write only outer edges
                                )
                            p_res(pX, pY) = *it;
                        it++;
                    }
                }
                shiftY += recvSizesY[rank] - 2;
            }
            shiftX += recvSizesX[rX] - 2;
        }

        std::ofstream file;
        char fname[1024];
        sprintf(fname, "result_%d_%d_%d.p", helper.getNumOfProcs(), grid.numOfPointsX, grid.numOfPointsY);
        file.open(fname);
        for (unsigned int y = 0; y < p_res.sizeY(); ++y) {
            for (unsigned int x = 0; x < p_res.sizeX(); ++x) {
                file << p_res(x, y) << " ";
                if (x != 0 && y != 0 && x != p_res.sizeX() - 1 && y != p_res.sizeY() - 1) {
                    double resDiff = p_res(x, y) - trueAnswer(grid.x[x], grid.y[y]);
                    solutionError += resDiff * resDiff * grid.x.getMidStep(x) * grid.y.getMidStep(y);
                }
            }
            file << std::endl;
        }
        file.close();
        solutionError = std::sqrt(solutionError);
    }
}

void ConjugateGradientMethod::collectResults() {
    collectP();
    MPIHelper &helper = MPIHelper::getInstance();

    if (helper.isMaster()) {
#ifdef USE_OMP
        printf("Parameters: Grid = (%d, %d); Processes = %d; OMP threads = %d\n",
               p.grid.numOfPointsX, p.grid.numOfPointsY, helper.getNumOfProcs(), helper.getNumOfOMPThreads());
#else
        printf("Parameters: Grid = (%d, %d); Processes = %d\n",
               p.grid.numOfPointsX, p.grid.numOfPointsY, helper.getNumOfProcs());
#endif
        printf("Elapsed time: %lf s.; Iterations = %d; Error = %lf\n", timeElapsed, numIters, solutionError);
    }
}
