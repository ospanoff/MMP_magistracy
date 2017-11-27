//
// Created by Ayat ".ospanoff" Ospanov
//

#ifndef FUNC2D_H
#define FUNC2D_H

#define CUDA_SAFE_CALL(call)\
do {\
    cudaError_t err = call;\
    if (cudaSuccess != err) {\
        char s[1024];\
        sprintf(s, "Cuda error in file '%s' in line %i: %s",\
        __FILE__, __LINE__, cudaGetErrorString(err));\
        throw std::string(s);\
    }\
} while (false)\


#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <limits>

#include "Grid.h"
#include "Exchanger.h"


typedef double (*mathFunction)(double, double);
typedef thrust::tuple<double, unsigned long> tuple;
typedef thrust::device_vector<double> vector;

static double zero(double, double) { return 0.0; }
static double one(double, double) { return 1.0; }
static double infty(double, double) { return std::numeric_limits<double>::max(); }


class Func2D {
public:
    Grid grid;
    double *fRaw;
    thrust::device_ptr<double> fDev;
    unsigned long pitch;
    unsigned long pitchDouble;

public:
    explicit Func2D(const Grid &grid, mathFunction func=zero);
    ~Func2D();

    unsigned int sizeX() const;
    unsigned int sizeY() const;

    bool operator>(double x) const;
    double operator()(unsigned int i, unsigned int j);
    void operator()(unsigned int i, unsigned int j, double val);
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
