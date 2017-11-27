//
// Created by Ayat ".ospanoff" Ospanov
//

#include <cmath>

#include "Func2D.h"
#include "MPIHelper.h"


Func2D::Func2D(const Grid &grid, mathFunction func, int initType)
        :grid(grid)
{

    CUDA_SAFE_CALL(cudaMallocPitch(&fRaw, &pitch, grid.x.size() * sizeof(double), grid.y.size()));

    pitchDouble = pitch / sizeof(double);
    fDev = thrust::device_pointer_cast(fRaw);

    pitchVecSize = grid.y.size() * pitchDouble;

    double *f = new double[pitchVecSize];
    std::fill(f, f + pitchVecSize, 0.);
    switch (initType) {
        case initAll: // shift = 0
        case initInner: { // shift = 1
            unsigned int shift = static_cast<unsigned int>(initType);
            for (unsigned int i = shift; i < grid.x.size() - shift; ++i) {
                for (unsigned int j = shift; j < grid.y.size() - shift; ++j) {
                    f[j * pitchDouble + i] = func(grid.x[i], grid.y[j]);
                }
            }
            break;
        }
        case initOuterBorder:
            MPIHelper &helper = MPIHelper::getInstance();
            if (!helper.hasLeftNeighbour()) {
                for (unsigned int j = 0; j < grid.y.size(); ++j) {
                    f[j * pitchDouble] = func(grid.x[0], grid.y[j]);
                }
            }
            if (!helper.hasRightNeighbour()) {
                unsigned int right = grid.x.size() - 1;
                for (unsigned int j = 0; j < grid.y.size(); ++j) {
                    f[j * pitchDouble + right] = func(grid.x[right], grid.y[j]);
                }
            }
            if (!helper.hasTopNeighbour()) {
                for (unsigned int i = 0; i < grid.x.size(); ++i) {
                    f[i] = func(grid.x[i], grid.y[0]);
                }
            }
            if (!helper.hasBottomNeighbour()) {
                unsigned int bottom = grid.y.size() - 1;
                for (unsigned int i = 0; i < grid.x.size(); ++i) {
                    f[bottom * pitchDouble + i] = func(grid.x[i], grid.y[bottom]);
                }
            }
            break;
    }
    CUDA_SAFE_CALL(cudaMemcpy2D(fRaw, pitch, f, pitch, grid.x.size() * sizeof(double), grid.y.size(), cudaMemcpyHostToDevice));
    delete[] f;
}

Func2D::~Func2D() {
    CUDA_SAFE_CALL(cudaFree(fRaw));
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

double Func2D::operator()(unsigned int i, unsigned int j) {
    return fDev[j * pitchDouble + i];
}

void Func2D::operator()(unsigned int i, unsigned int j, double val) {
    fDev[j * pitchDouble + i] = val;
}

struct gridMul {
    unsigned long pitchDouble, rBorder, bBorder;
    const double *midStepX, *midStepY;

    gridMul(const Grid &grid, unsigned long pitch)
            :pitchDouble(pitch),rBorder(grid.x.size()),bBorder(grid.y.size()),
             midStepX(grid.x.getMidStepPtr()),midStepY(grid.y.getMidStepPtr()) {}

    __host__ __device__
    double operator()(tuple valIdx, double val2) {
        unsigned long idx = thrust::get<1>(valIdx);
        unsigned long i = idx % pitchDouble;
        unsigned long j = idx / pitchDouble;
        if (i < 1 || i > rBorder - 2 || j < 1 || j > bBorder - 2) {
            return 0.;
        } else {
            return thrust::get<0>(valIdx) * val2 * midStepX[i - 1] * midStepY[j - 1];
        }
    }
};

double Func2D::operator*(const Func2D &func) const {
    thrust::counting_iterator<unsigned long> iter(0);
    double mul = thrust::inner_product(
            thrust::make_zip_iterator(thrust::make_tuple(fDev, iter)),
            thrust::make_zip_iterator(thrust::make_tuple(fDev + pitchVecSize, iter + pitchVecSize)),
            func.fDev,
            0.,
            thrust::plus<double>(),
            gridMul(grid, pitchDouble)
    );
    MPIHelper::getInstance().AllReduceSum(mul);
    return mul;
}

Func2D Func2D::operator*(double x) const {
    Func2D mul(grid);
    thrust::transform(
        fDev,
        fDev + pitchVecSize,
        mul.fDev,
        thrust::placeholders::_1 * x
    );
    return mul;
}

Func2D Func2D::operator-(const Func2D &func) const {
    Func2D diff(grid);
    thrust::transform(
        fDev,
        fDev + pitchVecSize,
        func.fDev,
        diff.fDev,
        thrust::minus<double>()
    );
    return diff;
}

Func2D &Func2D::operator-=(const Func2D &func) {
    thrust::transform(
        fDev,
        fDev + pitchVecSize,
        func.fDev,
        fDev,
        thrust::minus<double>()
    );
    return *this;
}

__global__
void gridLaplacianKernel(double *dest, double *src, const double *midStepX, const double *midStepY,
                         const double *stepX, const double *stepY,
                         unsigned long pitch, unsigned long rBorder, unsigned long bBorder) {
    unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && x < rBorder - 1 && y > 0 && y < bBorder - 1) {
        double *p_i_j = (double *) ((char *) src + y * pitch) + x;
        double *p_im1_j = p_i_j - 1;
        double *p_ip1_j = p_i_j + 1;
        double *p_i_jm1 = (double *) ((char *) src + (y - 1) * pitch) + x;
        double *p_i_jp1 = (double *) ((char *) src + (y + 1) * pitch) + x;
        double *new_p_i_j = (double *) ((char *) dest + y * pitch) + x;
        *new_p_i_j = ((*p_i_j - *p_im1_j) / stepX[x - 1] - (*p_ip1_j - *p_i_j) / stepX[x]) / midStepX[x - 1];
        *new_p_i_j += ((*p_i_j - *p_i_jm1) / stepY[y - 1] - (*p_i_jp1 - *p_i_j) / stepY[y]) / midStepY[y - 1];
    }
}

Func2D Func2D::operator~() const {
    Func2D laplace(grid);
    dim3 blockDim(16, 16);
    dim3 gridDim((grid.x.size() - 1) / blockDim.x + 1, (grid.y.size() - 1) / blockDim.y + 1);
    gridLaplacianKernel<<<gridDim, blockDim>>>(laplace.fRaw, fRaw, grid.x.getMidStepPtr(), grid.y.getMidStepPtr(),
                                          grid.x.getStepPtr(), grid.y.getStepPtr(),
                                          pitch, grid.x.size(), grid.y.size());
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    return laplace;
}

Func2D &Func2D::operator=(const Func2D &func) {
    CUDA_SAFE_CALL(cudaMemcpy2D(fRaw, pitch, func.fRaw, func.pitch, grid.x.size() * sizeof(double), grid.y.size(), cudaMemcpyDeviceToDevice));
    return *this;
}

void Func2D::synchronize(std::vector<Exchanger> &communicatingEdges) {
    double *f = new double[pitchVecSize];
    CUDA_SAFE_CALL(cudaMemcpy2D(f, pitch, fRaw, pitch, grid.x.size() * sizeof(double), grid.y.size(), cudaMemcpyDeviceToHost));

    for (std::vector<Exchanger>::iterator it = communicatingEdges.begin(); it != communicatingEdges.end(); ++it) {
        exchange(*it, f);
    }
    for (std::vector<Exchanger>::iterator it = communicatingEdges.begin(); it != communicatingEdges.end(); ++it) {
        wait(*it, f);
    }

    CUDA_SAFE_CALL(cudaMemcpy2D(fRaw, pitch, f, pitch, grid.x.size() * sizeof(double), grid.y.size(), cudaMemcpyHostToDevice));
    delete[] f;
}

void Func2D::exchange(Exchanger &exchanger, double *f) {
    MPIHelper &helper = MPIHelper::getInstance();
    exchanger.sendData.clear();
    for (int i = exchanger.sendPart.startX; i < exchanger.sendPart.endX; ++i) {
        for (int j = exchanger.sendPart.startY; j < exchanger.sendPart.endY; ++j) {
            exchanger.sendData.push_back(f[j * pitchDouble + i]);
        }
    }

    helper.Isend(exchanger.sendData, exchanger.exchangeRank, exchanger.sendReq);
    helper.Irecv(exchanger.recvData, exchanger.exchangeRank, exchanger.recvReq);
}

void Func2D::wait(Exchanger &exchanger, double *f) {
    MPIHelper &helper = MPIHelper::getInstance();
    helper.Wait(exchanger.recvReq);

    std::vector<double>::const_iterator value = exchanger.recvData.begin();
    for (int i = exchanger.recvPart.startX; i < exchanger.recvPart.endX; ++i) {
        for (int j = exchanger.recvPart.startY; j < exchanger.recvPart.endY; ++j) {
            f[j * pitchDouble + i] = *(value++);
        }
    }

    helper.Wait(exchanger.sendReq);
}

void Func2D::print() {
    for (int j = 0; j < grid.y.size(); ++j) {
        for (int i = 0; i < grid.x.size(); ++i) {
            std::cout << fDev[j * pitchDouble + i] << " ";
        }
        std::cout << std::endl;
    }
}