//
// Created by Ayat ".ospanoff" Ospanov
//

#ifndef CGM_H
#define CGM_H


#include "Func2D.h"
#include "Exchanger.h"


class ConjugateGradientMethod {
    Func2D p, r, g, F, diff;
    double eps;
    double diffNorm;
    mathFunction trueAnswer;

    bool display;

    std::vector<Exchanger> communicatingEdges;

    int numIters;
    double timeElapsed;
    double solutionError;

    double tau(const Func2D &f, const Func2D &g) const;
    double alpha(const Func2D &f, const Func2D &g) const;
public:
    ConjugateGradientMethod(Grid grid, mathFunction f, mathFunction phi,
                            double eps, bool display, mathFunction answer=zero);
    void solve();
    void initialStep();
    void iteration();

    void collectP();
    void collectResults();
};


#endif //CGM_H
