//
// Created by Ayat ".ospanoff" Ospanov
//

#ifndef GRID_H
#define GRID_H


#include <vector>


template <class T>
struct Rect {
    T A1, A2, B1, B2;
    Rect(T A1, T A2, T B1, T B2) :
            A1(A1), A2(A2), B1(B1), B2(B2) {}
};


class Grid1D {
    double A1;
    double A2;
    unsigned int numOfPts;

    std::vector<double> grid;
    std::vector<double> step;
    std::vector<double> midStep;

public:
    Grid1D(double A1, double A2, unsigned int begin, unsigned int end, unsigned int numOfPts);

    double f(double t, float q=3.f/2);
    double getCoord(unsigned int i);

    double &operator[](unsigned int i) {
        return grid[i];
    }
    double operator[](unsigned int i) const {
        return grid[i];
    }
    unsigned int size() const {
        return static_cast<unsigned int>(grid.size());
    }

    double getStep(unsigned int i) const;
    double getMidStep(unsigned int i) const;
};

class Grid {
public:
    Grid1D x, y;
    Rect<double> borders;
    Rect<unsigned int> edges;
    unsigned int numOfPointsX, numOfPointsY;

public:
    Grid(Rect<double> borders, Rect<unsigned int> edges, unsigned int numX, unsigned int numY)
            :borders(borders),edges(edges),
             numOfPointsX(numX),numOfPointsY(numY),
             x(borders.A1, borders.A2, edges.A1, edges.A2, numX),
             y(borders.B1, borders.B2, edges.B1, edges.B2, numY)
    {}

    static void countEdges(unsigned int numOfPts, int numOfBlocks, int blockIdx, unsigned int &begin,
                           unsigned int &end);
    static Grid getGrid(Rect<double> borders, unsigned int numOfPointsX, unsigned int numOfPointsY);
};


#endif //GRID_H
