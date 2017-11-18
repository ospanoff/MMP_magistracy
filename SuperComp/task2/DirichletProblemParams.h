//
// Created by Ayat ".ospanoff" Ospanov
//

#ifndef DIRICHLET_PROBLEM_PARAMS_H
#define DIRICHLET_PROBLEM_PARAMS_H

#include <cmath>

#include "ConjugateGradientMethod.h"

class DirichletProblemParams : public Problem {
public:
    DirichletProblemParams() {
        A1 = 0.0;
        A2 = 2.0;
        B1 = 0.0;
        B2 = 2.0;
        eps = 0.0001;
    }
    inline double Laplacian(const Func2D &func, const Grid1D &gridX, const Grid1D &gridY, int x, int y) {
        double fr1 = (func(x, y) - func(x - 1, y)) / gridX.step(x - 1);
        double fr2 = (func(x + 1, y) - func(x, y)) / gridX.step(x);
        double L = (fr1 - fr2) / gridX.midStep(x);

        fr1 = (func(x, y) - func(x, y - 1)) / gridY.step(y - 1);
        fr2 = (func(x, y + 1) - func(x, y)) / gridY.step(y);
        L += (fr1 - fr2) / gridY.midStep(y);

        return L;
    }

    inline double F(double x, double y) const {
        double tmp = 1.0 + x * x + y * y;
        return 8.0 * (1.0 - x * x - y * y) / (tmp * tmp * tmp);
    }
    inline double phi(double x, double y) const {
        return 2.0 / (1.0 + x * x + y * y);
    }

    inline void computeR(const Func2D &p, Func2D &r, const Grid1D &gridX, const Grid1D &gridY) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int x = 1; x < r.sizeX - 1; ++x) {
            for (int y = 1; y < r.sizeY - 1; ++y) {
                r(x, y) = Laplacian(p, gridX, gridY, x, y) - F(gridX[x], gridY[y]);
            }
        }
    };

    inline void computeG(double alpha, const Func2D &r, Func2D &g) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int x = 1; x < g.sizeX - 1; ++x) {
            for (int y = 1; y < g.sizeY - 1; ++y) {
                g(x, y) = r(x, y) - alpha * g(x, y);
            }
        }
    };

    inline double computeP(double tau, const Func2D &g, Func2D &p, const Grid1D &gridX, const Grid1D &gridY) {
        double diff = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+:diff)
#endif
        for (int x = 1; x < p.sizeX - 1; ++x) {
            for (int y = 1; y < p.sizeY - 1; ++y) {
                const double p_diff = tau * g(x, y);
                diff += p_diff * p_diff * gridX.midStep(x) * gridY.midStep(y);
                p(x, y) = p(x, y) - p_diff;
            }
        }
        return diff;
    };

    inline Fraction computeTau(const Func2D &r, const Func2D &g, const Grid1D &gridX, const Grid1D &gridY) {
        double numerator = 0;
        double denominator = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+:numerator, denominator)
#endif
        for (int x = 1; x < g.sizeX - 1; ++x) {
            for (int y = 1; y < g.sizeY - 1; ++y) {
                const double g_step = g(x, y) * gridX.midStep(x) * gridY.midStep(y);
                numerator += r(x, y) * g_step;
                denominator += Laplacian(g, gridX, gridY, x, y) * g_step;
            }
        }
        return Fraction(numerator, denominator);
    };

    inline Fraction computeAlpha(const Func2D &r, const Func2D &g, const Grid1D &gridX, const Grid1D &gridY) {
        double numerator = 0;
        double denominator = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+:numerator, denominator)
#endif
        for (int x = 1; x < g.sizeX - 1; ++x) {
            for (int y = 1; y < g.sizeY - 1; ++y) {
                const double g_step = g(x, y) * gridX.midStep(x) * gridY.midStep(y);
                numerator += Laplacian(r, gridX, gridY, x, y) * g_step;
                denominator += Laplacian(g, gridX, gridY, x, y) * g_step;
            }
        }
        return Fraction(numerator, denominator);
    };
};


#endif //DIRICHLET_PROBLEM_PARAMS_H
