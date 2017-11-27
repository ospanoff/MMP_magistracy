//
// Created by Ayat ".ospanoff" Ospanov
//

#include <cmath>
#include "Grid.h"
#include "MPIHelper.h"
#include "Func2D.h"


////////////////////////////////////////////////////////////////////////////////
/// Grid1D
////////////////////////////////////////////////////////////////////////////////
Grid1D::Grid1D(double A1, double A2, unsigned int begin, unsigned int end, unsigned int numOfPts)
        :A1(A1),A2(A2),numOfPts(numOfPts)
{
    grid.reserve(end - begin);
    for (unsigned int i = begin; i < end; ++i) {
        grid.push_back(getCoord(i));
    }

    step.reserve(end - begin - 1);
    for (unsigned int i = 0; i < end - begin - 1; ++i) {
        step.push_back(grid[i + 1] - grid[i]);
    }

    midStep.reserve(end - begin - 2);
    for (unsigned int i = 1; i < end - begin - 1; ++i) {
        midStep.push_back(0.5 * (step[i] + step[i - 1]));
    }

    stepDev.resize(step.size());
    thrust::copy(step.begin(), step.end(), stepDev.begin());

    midStepDev.resize(midStep.size());
    thrust::copy(midStep.begin(), midStep.end(), midStepDev.begin());
}

double Grid1D::f(double t, float q) {
    return (std::pow(static_cast<float>(1.0 + t), q) - 1.0) / (std::pow(2.0f, q) - 1.0);
}

double Grid1D::getCoord(unsigned int i) {
    double t = 1.0 * i / (numOfPts - 1);
    return A2 * f(t) + A1 * (1 - f(t));
}

double Grid1D::getStep(unsigned int i) const {
    return step[i];
}

double Grid1D::getMidStep(unsigned int i) const {
    return midStep[i - 1];
}

const double * Grid1D::getStepPtr() const {
    return thrust::raw_pointer_cast(stepDev.data());
}
const double * Grid1D::getMidStepPtr() const {
    return thrust::raw_pointer_cast(midStepDev.data());
}


////////////////////////////////////////////////////////////////////////////////
/// Grid
////////////////////////////////////////////////////////////////////////////////
void Grid::countEdges(unsigned int numOfPts, int numOfBlocks, int blockIdx, unsigned int &begin, unsigned int &end) {
    int blockSize = numOfPts / numOfBlocks;
    int extraPts = numOfPts % numOfBlocks;

    // We are adding extra pts 1by1 to each block, so next blocks should have shifts
    unsigned int begin_shift = static_cast<unsigned int>(std::min(blockIdx, extraPts));
    unsigned int end_shift = blockIdx < extraPts ? 1 : 0;

    begin = blockIdx * blockSize + begin_shift;
    end = begin + blockSize + end_shift;
}

Grid Grid::getGrid(Rect<double> borders, unsigned int numOfPointsX, unsigned int numOfPointsY) {
    MPIHelper &helper = MPIHelper::getInstance();
    unsigned int beginEdgeX, endEdgeX, beginEdgeY, endEdgeY;
    countEdges(numOfPointsX, helper.getNumOfProcsX(), helper.getRankX(), beginEdgeX, endEdgeX);
    countEdges(numOfPointsY, helper.getNumOfProcsY(), helper.getRankY(), beginEdgeY, endEdgeY);
    if (helper.hasLeftNeighbour())
        beginEdgeX--;
    if (helper.hasRightNeighbour())
        endEdgeX++;
    if (helper.hasTopNeighbour())
        beginEdgeY--;
    if (helper.hasBottomNeighbour())
        endEdgeY++;
    return Grid(borders, Rect<unsigned int>(beginEdgeX, endEdgeX, beginEdgeY, endEdgeY), numOfPointsX, numOfPointsY);
}
