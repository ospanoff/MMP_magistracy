//
// Created by Ayat ".ospanoff" Ospanov
//

#ifndef DirichletProblem_H
#define DirichletProblem_H


#include "Grid.h"


struct DirichletProblem {
    static Rect<double> getBorders() {
        return Rect<double>(0.0, 2.0, 0.0, 2.0);
    }

    static double F(double x, double y) {
        double tmp = 1.0 + x * x + y * y;
        return 8.0 * (1.0 - x * x - y * y) / (tmp * tmp * tmp);
    }

    static double phi(double x, double y) {
        return 2.0 / (1.0 + x * x + y * y);
    }

    static double answer(double x, double y) {
        return phi(x, y);
    }
};


#endif //DirichletProblem_H
