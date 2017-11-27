//
// Created by Ayat ".ospanoff" Ospanov
//

#include <cmath>
#include "Func2D.h"
#include "MPIHelper.h"


Func2D::Func2D(const Grid &grid, mathFunction func)
        :grid(grid)
{
    hacked = false;
    f = new double*[grid.x.size()];
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (unsigned int i = 0; i < grid.x.size(); ++i) {
        f[i] = new double[grid.y.size()];
        for (unsigned int j = 0; j < grid.y.size(); ++j) {
            if (i != 0 && j != 0 && i != grid.x.size() - 1 && j != grid.y.size() - 1) {
                f[i][j] = func(grid.x[i], grid.y[j]);
            } else {
                f[i][j] = 0;
            }
        }
    }
}

Func2D::~Func2D() {
    for (unsigned int i = 0; i < grid.x.size(); ++i) {
        delete[] f[i];
    }
    delete[] f;
}

unsigned int Func2D::sizeX() const {
    return grid.x.size();
}

unsigned int Func2D::sizeY() const {
    return grid.y.size();
}

bool Func2D::operator>(double x) const {
    return std::sqrt(*this * *this) > x;
}

double &Func2D::operator()(unsigned int i, unsigned int j) {
    return f[i][j];
}

double Func2D::operator*(const Func2D &func) const {
    double mul = 0;
#ifdef USE_OMP
    #pragma omp parallel for reduction(+:mul)
#endif
    for (unsigned int i = 1; i < grid.x.size() - 1; ++i) {
        for (unsigned int j = 1; j < grid.y.size() - 1; ++j) {
            mul += f[i][j] * func.f[i][j] * grid.x.getMidStep(i) * grid.y.getMidStep(j);
        }
    }
    MPIHelper::getInstance().AllReduceSum(mul);
    return mul;
}

Func2D Func2D::operator*(double x) const {
    Func2D mul(grid);
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (unsigned int i = 1; i < grid.x.size() - 1; ++i) {
        for (unsigned int j = 1; j < grid.y.size() - 1; ++j) {
            mul.f[i][j] = x * f[i][j];
        }
    }
    return mul;
}

Func2D Func2D::operator-(const Func2D &func) const {
    Func2D diff(grid);
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (unsigned int i = 1; i < grid.x.size() - 1; ++i) {
        for (unsigned int j = 1; j < grid.y.size() - 1; ++j) {
            diff.f[i][j] = f[i][j] - func.f[i][j];
        }
    }
    return diff;
}

Func2D &Func2D::operator-=(const Func2D &func) {
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (unsigned int i = 1; i < grid.x.size() - 1; ++i) {
        for (unsigned int j = 1; j < grid.y.size() - 1; ++j) {
            f[i][j] -= func.f[i][j];
        }
    }
    return *this;
}

Func2D Func2D::operator~() const {
    Func2D laplace(grid);
    if (!hacked) {
#ifdef USE_OMP
        #pragma omp parallel for
#endif
        for (unsigned int i = 1; i < grid.x.size() - 1; ++i) {
            for (unsigned int j = 1; j < grid.y.size() - 1; ++j) {
                laplace.f[i][j] = (
                        (
                                (f[i][j] - f[i - 1][j]) / grid.x.getStep(i - 1) -
                                (f[i + 1][j] - f[i][j]) / grid.x.getStep(i)
                        ) / grid.x.getMidStep(i) +
                        (
                                (f[i][j] - f[i][j - 1]) / grid.y.getStep(j - 1) -
                                (f[i][j + 1] - f[i][j]) / grid.y.getStep(j)
                        ) / grid.y.getMidStep(j)
                );
            }
        }
    } else {
        double stepX = 1.0 / grid.x.getMidStep(1);
        double stepY = 1.0 / grid.y.getMidStep(1);
        stepX *= stepX;
        stepY *= stepY;
#ifdef USE_OMP
        #pragma omp parallel for
#endif
        for (unsigned int i = 1; i < grid.x.size() - 1; ++i) {
            for (unsigned int j = 1; j < grid.y.size() - 1; ++j) {
                laplace.f[i][j] = stepX * (2.0 * f[i][j] - f[i - 1][j] - f[i + 1][j]) +
                        stepY * (2.0 * f[i][j] - f[i][j - 1] - f[i][j + 1]);
            }
        }
    }
    return laplace;
}

Func2D &Func2D::operator=(const Func2D &func) {
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (unsigned int i = 0; i < func.sizeX(); ++i) {
        for (unsigned int j = 0; j < func.sizeY(); ++j) {
            f[i][j] = func.f[i][j];
        }
    }
    return *this;
}

void Func2D::synchronize(std::vector<Exchanger> communicatingEdges) {
    for (std::vector<Exchanger>::iterator it = communicatingEdges.begin(); it != communicatingEdges.end(); ++it) {
        exchange(*it);
    }
    for (std::vector<Exchanger>::iterator it = communicatingEdges.begin(); it != communicatingEdges.end(); ++it) {
        wait(*it);
    }
}

void Func2D::exchange(Exchanger &exchanger) {
    MPIHelper &helper = MPIHelper::getInstance();
    exchanger.sendData.clear();
    for (int i = exchanger.sendPart.startX; i < exchanger.sendPart.endX; ++i) {
        for (int j = exchanger.sendPart.startY; j < exchanger.sendPart.endY; ++j) {
            exchanger.sendData.push_back(f[i][j]);
        }
    }

    helper.Isend(exchanger.sendData, exchanger.exchangeRank, exchanger.sendReq);
    helper.Irecv(exchanger.recvData, exchanger.exchangeRank, exchanger.recvReq);
}

void Func2D::wait(Exchanger &exchanger) {
    MPIHelper &helper = MPIHelper::getInstance();
    helper.Wait(exchanger.recvReq);

    std::vector<double>::const_iterator value = exchanger.recvData.begin();
    for (int i = exchanger.recvPart.startX; i < exchanger.recvPart.endX; ++i) {
        for (int j = exchanger.recvPart.startY; j < exchanger.recvPart.endY; ++j) {
            f[i][j] = *(value++);
        }
    }

    helper.Wait(exchanger.sendReq);
}
