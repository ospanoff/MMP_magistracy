//
// Created by Ayat ".ospanoff" Ospanov
//

#ifndef FUNC2D_H
#define FUNC2D_H


#include <limits>

#include "Grid.h"
#include "Exchanger.h"


typedef double (*mathFunction)(double, double);

static double zero(double, double) { return 0.0; }
static double infty(double, double) { return std::numeric_limits<double>::max(); }


class Func2D {
public:
    Grid grid;
    double **f;
    bool hacked;

public:
    explicit Func2D(const Grid &grid, mathFunction func=zero);
    ~Func2D();

    unsigned int sizeX() const;
    unsigned int sizeY() const;


    bool operator>(double x) const;
    double &operator()(unsigned int i, unsigned int j);
    double operator*(const Func2D &func) const;
    Func2D operator*(double x) const;
    Func2D operator-(const Func2D &func) const;
    Func2D operator~() const;
    Func2D &operator-=(const Func2D &func);
    Func2D &operator=(const Func2D &func);

    void synchronize(std::vector<Exchanger> communicatingEdges);
    void exchange(Exchanger &);
    void wait(Exchanger &);

};


#endif //FUNC2D_H
