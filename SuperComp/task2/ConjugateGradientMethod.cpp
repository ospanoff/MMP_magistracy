//
// Created by Ayat ".ospanoff" Ospanov
//

#include <cmath>
#include <iostream>
#include <fstream>

#include "ConjugateGradientMethod.h"


////////////////////////////////////////////////////////////////////////////////
/// ConjugateGradientMethod
////////////////////////////////////////////////////////////////////////////////
void ConjugateGradientMethod::count_set_processes() {
    int powX = 0, powY = 0;
    int Nproc = 1;
    double NX = numOfPointsX;
    double NY = numOfPointsY;

    while (Nproc < helper.numOfProcesses) {
        Nproc *= 2;
        if (NX > NY) {
            NX /= 2;
            powX++;
        } else {
            NY /= 2;
            powY++;
        }
    }

    if (Nproc != helper.numOfProcesses) {
        throw CGMException("Number of processes should be the power of 2");
    }

    numOfProcX = 1 << powX;
    numOfProcY = 1 << powY;
}

void ConjugateGradientMethod::count_edges(int numOfPts, int numOfBlocks, int blockIdx, int &begin, int &end) {
    int blockSize = numOfPts / numOfBlocks;
    int extraPts = numOfPts % numOfBlocks;

    // We are adding extra pts 1by1 to each block, so next blocks should have shifts
    int begin_shift = std::min(blockIdx, extraPts);
    int end_shift = blockIdx < extraPts ? 1 : 0;

    begin = blockIdx * blockSize + begin_shift;
    end = begin + blockSize + end_shift;
}

void ConjugateGradientMethod::set_edges() {
    rankX = helper.rank % numOfProcX;
    rankY = helper.rank / numOfProcX;
    count_edges(numOfPointsX, numOfProcX, rankX, beginEdgeX, endEdgeX);
    count_edges(numOfPointsY, numOfProcY, rankY, beginEdgeY, endEdgeY);
    if (has_left_neighbour())
        beginEdgeX--;
    if (has_right_neighbour())
        endEdgeX++;
    if (has_top_neighbour())
        beginEdgeY--;
    if (has_bottom_neighbour())
        endEdgeY++;
}

void ConjugateGradientMethod::init() {
    count_set_processes();
    set_edges();
    gridX.init(beginEdgeX, endEdgeX);
    gridY.init(beginEdgeY, endEdgeY);
    if (has_left_neighbour()) {
        communication.push_back(
                Exchanger(helper, get_1D_rank(rankX - 1, rankY),
                          EdgeIndexer(EdgeIndexer::ByX, 1, gridY),
                          EdgeIndexer(EdgeIndexer::ByX, 0, gridY)
                )
        );
    }
    if (has_right_neighbour()) {
        communication.push_back(
                Exchanger(helper, get_1D_rank(rankX + 1, rankY),
                          EdgeIndexer(EdgeIndexer::ByX, gridX.size() - 2, gridY),
                          EdgeIndexer(EdgeIndexer::ByX, gridX.size() - 1, gridY)
                )
        );
    }
    if (has_top_neighbour()) {
        communication.push_back(
                Exchanger(helper, get_1D_rank(rankX, rankY - 1),
                          EdgeIndexer(EdgeIndexer::ByY, 1, gridX),
                          EdgeIndexer(EdgeIndexer::ByY, 0, gridX)
                )
        );
    }
    if (has_bottom_neighbour()) {
        communication.push_back(
                Exchanger(helper, get_1D_rank(rankX, rankY + 1),
                          EdgeIndexer(EdgeIndexer::ByY, gridY.size() - 2, gridX),
                          EdgeIndexer(EdgeIndexer::ByY, gridY.size() - 1, gridX)
                )
        );
    }
}

void ConjugateGradientMethod::communicate(Func2D &data) {
    for (std::vector<Exchanger>::iterator it = communication.begin(); it != communication.end(); ++it) {
        it->exchange(data);
    }
    for (std::vector<Exchanger>::iterator it = communication.begin(); it != communication.end(); ++it) {
        it->wait(data);
    }
}

void ConjugateGradientMethod::collectFraction(Fraction &f) {
    std::vector<double> d;
    d.push_back(f.numerator);
    d.push_back(f.denominator);
    helper.AllReduce(d);
    f.numerator = d[0];
    f.denominator = d[1];
}

void ConjugateGradientMethod::collectDouble(double &num) {
    std::vector<double> d;
    d.push_back(num);
    helper.AllReduce(d);
    num = d[0];
}

void ConjugateGradientMethod::solve() {
    double start = helper.Time();
    initialStep();
    while (diff > problem.eps) {
        numIter++;
        iteration();
    }
    timeElapsed = helper.Time() - start;
}

void ConjugateGradientMethod::initialStep() {
    /// Compute p0
    p.resize(gridX.size(), gridY.size());

    if (!has_left_neighbour()) {
        for (int y = 0; y < p.sizeY; ++y) {
            p(0, y) = problem.phi(gridX[0], gridY[y]);
        }
    }
    if (!has_right_neighbour()) {
        int right = p.sizeX - 1;
        for (int y = 0; y < p.sizeY; ++y) {
            p(right, y) = problem.phi(gridX[right], gridY[y]);
        }
    }
    if (!has_top_neighbour()) {
        for (int x = 0; x < p.sizeX; ++x) {
            p(x, 0) = problem.phi(gridX[x], gridY[0]);
        }
    }
    if (!has_bottom_neighbour()) {
        int bottom = p.sizeY - 1;
        for (int x = 0; x < p.sizeX; ++x) {
            p(x, bottom) = problem.phi(gridX[x], gridY[bottom]);
        }
    }

    /// Compute r0
    r.resize(gridX.size(), gridY.size());
    problem.computeR(p, r, gridX, gridY);
    communicate(r);

    /// Compute g0
    g = r;

    /// Compute tau1
    Fraction tau = problem.computeTau(r, g, gridX, gridY);
    collectFraction(tau);

    /// Compute p1
    diff = problem.computeP(tau.divide(), g, p, gridX, gridY);
    collectDouble(diff);
}

void ConjugateGradientMethod::iteration() {
    /// k = 1, 2, ...
    communicate(p);

    /// r_{k} ~ p_{k}
    problem.computeR(p, r, gridX, gridY);
    communicate(r);

    /// alpha_{k} ~ r_{k}, g_{k-1}
    Fraction alpha = problem.computeAlpha(r, g, gridX, gridY);
    collectFraction(alpha);

    /// g_{k} ~ alpha_{k}
    problem.computeG(alpha.divide(), r, g);
    communicate(g);

    /// tau_{k+1} ~ r_{k}, g_{k}
    Fraction tau = problem.computeTau(r, g, gridX, gridY);
    collectFraction(tau);

    /// p_{k+1} = p_{k} - tau_{k+1} * g_{k}, k = 1, 2, ...
    diff = problem.computeP(tau.divide(), g, p, gridX, gridY);
    collectDouble(diff);
}

void ConjugateGradientMethod::collectP() {
    communicate(p);

    std::vector<double> recv;
    std::vector<int> recv_sizes;
    std::vector<int> recv_sizesX;
    std::vector<int> recv_sizesY;
    std::vector<int> recv_displ;

    if (helper.rank == 0) {
        recv_sizesX.resize(static_cast<unsigned int>(helper.numOfProcesses));
        recv_sizesY.resize(static_cast<unsigned int>(helper.numOfProcesses));
        recv_displ.resize(static_cast<unsigned int>(helper.numOfProcesses), 0);
    }

    helper.Gather(p.sizeX, recv_sizesX);
    helper.Gather(p.sizeY, recv_sizesY);

    if (helper.rank == 0) {
        recv_sizes.push_back(recv_sizesX[0] * recv_sizesY[0]);
        for (int i = 1; i < recv_sizesX.size(); ++i) {
            recv_sizes.push_back(recv_sizesX[i] * recv_sizesY[i]);
            recv_displ[i] = recv_displ[i - 1] + recv_sizes[i - 1];
        }
        int k = static_cast<int>(recv_displ.size() - 1);
        recv.resize(static_cast<unsigned int>(recv_displ[k] + recv_sizes[k]));
    }

    helper.Gatherv(p.f, recv, recv_sizes, recv_displ);

    if (helper.rank == 0) {
        Func2D p_res(numOfPointsX, numOfPointsY);

        int shiftX = 0;
        for (int rX = 0; rX < numOfProcX; ++rX) {
            int shiftY = 0;
            for (int rY = 0; rY < numOfProcY; ++rY) {
                int rank = get_1D_rank(rX, rY);
                std::vector<double>::const_iterator it = recv.begin() + recv_displ[rank];
                for (int y = 0; y < recv_sizesY[rank]; ++y) {
                    for (int x = 0; x < recv_sizesX[rank]; ++x) {
                        int pX = shiftX + x;
                        int pY = shiftY + y;
                        if (
                                (x != 0 && y != 0 && x != recv_sizesX[rank] - 1 && y != recv_sizesY[rank] - 1) || /// Don't write all edges
                                (pX == 0 || pX == numOfPointsX - 1 || pY == 0 || pY == numOfPointsY - 1) /// write only outer edges
                            )
                            p_res(pX, pY) = *it;
                        it++;
                    }
                }
                shiftY += recv_sizesY[rank] - 2;
            }
            shiftX += recv_sizesX[rX] - 2;
        }

        std::ofstream file;
        std::string fname = "result_" + to_string(helper.numOfProcesses) + "_" + to_string(numOfPointsX) + "_" + to_string(numOfPointsY) + ".p";
        file.open(fname.c_str());
        for (int y = 0; y < p_res.sizeY; ++y) {
            for (int x = 0; x < p_res.sizeX; ++x) {
                file << p_res(x, y) << " ";
            }
            file << std::endl;
        }
        file.close();
    }
}

void ConjugateGradientMethod::collectResults() {
    collectP();

    helper.ReduceMax(timeElapsed, timeElapsed);
    if (helper.rank == 0) {
#ifdef USE_OMP
        printf("Parameters: Grid = (%d, %d); Processes = %d; OMP threads = %d\n",
               numOfPointsX, numOfPointsY, helper.numOfProcesses, helper.numOfOMPThreads);
#else
        printf("Parameters: Grid = (%d, %d); Processes = %d\n",
               numOfPointsX, numOfPointsY, helper.numOfProcesses);
#endif
        printf("Elapsed time: %lf s.; Iterations = %d\n", timeElapsed, numIter);
    }
}
