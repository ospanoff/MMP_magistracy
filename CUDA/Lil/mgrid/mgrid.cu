#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#define  Max(a, b) ((a)>(b)?(a):(b))

#define CUDA_SAFE_CALL(call)\
do {\
    cudaError_t err = call;\
    if (cudaSuccess != err) {\
        printf("Cuda error in file '%s' in line %i: %s\n",\
        __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(1);\
    }\
} while (false)\

FILE *in;
int TRACE = 0;
double EPS;
int     M, N, K, ITMAX;
double  MAXEPS = 0.0000001;

double *A, *A_GPU;

int devCount = 0;

#define A(i,j,k) A[((i)*N+(j))*K+(k)]
#define a(i,j,k) a[((i)*nn+(j))*kk+(k)]
#define a2(i,j,k) a2[((i)*n2+(j))*k2+(k)]
#define tmp(d,i,j,k) tmp[d][((i)*nn+(j))*kk+(k)]

double solution(int i, int j, int k)
{
    double x = 10.*i / (M - 1), y = 10.*j / (N - 1), z = 10.*k / (K - 1);
    return 2.*x*x - y*y - z*z;
}

double jac(double *a, int mm, int nn, int kk, int itmax, double maxeps);

int main(int an, char **as)
{
    int i, j, k;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&devCount));

    in = fopen("data3.in", "r");
    if (in == NULL) { printf("Can not open 'data3.in' "); exit(1); }
    i = fscanf(in, "%d %d %d %d %d", &M, &N, &K, &ITMAX, &TRACE);
    if (i < 4)
    {
        printf("Wrong 'data3.in' (M N K ITMAX TRACE)");
        exit(2);
    }

    A = (double*) malloc(M*N*K * sizeof(double));

    for (i = 0; i <= M - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= K - 1; k++) {
                if (i == 0 || i == M - 1 || j == 0 || j == N - 1 || k == 0 || k == K - 1)
                    A(i, j, k) = solution(i, j, k);
                else 
                    A(i, j, k) = 0.;
            }

    CUDA_SAFE_CALL(cudaMalloc(&A_GPU, M*N*K * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpy(A_GPU, A, M*N*K * sizeof(double), cudaMemcpyHostToDevice));
    clock_t t = clock();

    EPS = jac(A_GPU, M, N, K, ITMAX, MAXEPS);

    t = clock() - t;
    double elapsed = 1.0 * t / CLOCKS_PER_SEC;
    printf("%dx%dx%d x %d\t<", M, N, K, ITMAX);
    printf("%3.5f s.>\teps=%.4g\n", elapsed, EPS);

    CUDA_SAFE_CALL(cudaMemcpy(A, A_GPU, M*N*K * sizeof(double), cudaMemcpyDeviceToHost));
    
    if (TRACE)
    {
        EPS = 0.;
        for (i = 0; i <= M - 1; i++)
            for (j = 0; j <= N - 1; j++)
                for (k = 0; k <= K - 1; k++)
                    EPS = Max(fabs(A(i, j, k) - solution(i, j, k)), EPS);
        printf("delta=%.4g\n", EPS);
    }

    free(A);
    CUDA_SAFE_CALL(cudaFree(A_GPU));
    return 0;
}

#define diff(i,j,k) diff[((i)*(nn - 2)+(j))*(kk - 2)+(k)]
#define border(j,k) border[(j)*kk+(k)]

__global__
void jac_kernel(double *a, int mm, int nn, int kk, double *diff, double *border, bool isLeft) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= 1 && i <= mm - 2 && j >= 1 && j <= nn - 2 && k >= 1 && k <= kk - 2) {
        double tmp = (a(i - 1, j, k) + a(i + 1, j, k) + a(i, j - 1, k) + a(i, j + 1, k)
                      + a(i, j, k - 1) + a(i, j, k + 1)) / 6.;
        diff(i - 1, j - 1, k - 1) = fabsf(a(i, j, k) - tmp);
        a(i, j, k) = tmp;
        if (isLeft && i == mm - 2) {
            border(j, k) = tmp;
        } else if (!isLeft && i == 1) {
            border(j, k) = tmp;
        }
    }
}

void run_jac_kernel(double *a, int mm, int nn, int kk, double *diff, double *border, bool isLeft) {
    dim3 gridDim = dim3((kk + 31) / 32, (nn + 31) / 32, mm);
    dim3 blockDim = dim3(32, 32, 1);
    jac_kernel<<<gridDim, blockDim>>>(a, mm, nn, kk, diff, border, isLeft);
    CUDA_SAFE_CALL(cudaGetLastError());
}


__global__
void writeBorder_kernel(double *a, int nn, int kk, double *border, int borderIdx) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = borderIdx;
    if (j >= 0 && j <= nn - 2 && k >= 1 && k <= kk - 2) {
        a(i, j, k) = border(j, k);
    }
}

void writeBorder(double *a, int nn, int kk, double *border, int borderIdx) {
    dim3 gridDim = dim3((kk + 31) / 32, (nn + 31) / 32);
    dim3 blockDim = dim3(32, 32);
    writeBorder_kernel<<<gridDim, blockDim>>>(a, nn, kk, border, borderIdx);
    CUDA_SAFE_CALL(cudaGetLastError());
}

__global__
void jac_kernel_inner1(double *a, int mm, int nn, int kk, double *a2, int n2, int k2) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int mm_ = mm - 2 * (1 - mm % 2);
    int nn_ = nn - 2 * (1 - nn % 2);
    int kk_ = kk - 2 * (1 - kk % 2);
    if (i <= mm_ - 1 && j <= nn_ - 1 && k <= kk_ - 1) {
        if (i % 2 == 0 && j % 2 == 0 && k % 2 == 0) {
            a2(i / 2, j / 2, k / 2) = a(i, j, k);
        }
    } else if (i <= mm - 1 && j <= nn - 1 && k <= kk - 1) {
        if (i == mm - 1 || j == nn - 1 || k == kk - 1) {
            a2(i / 2, j / 2, k / 2) = a(i, j, k);
        }
    }
}

void run_jac_kernel_inner1(double *a, int mm, int nn, int kk, double *a2, int n2, int k2) {
    dim3 gridDim = dim3((kk + 31) / 32, (nn + 31) / 32, mm);
    dim3 blockDim = dim3(32, 32, 1);
    jac_kernel_inner1<<<gridDim, blockDim>>>(a, mm, nn, kk, a2, n2, k2);
    CUDA_SAFE_CALL(cudaGetLastError());
}

__global__
void jac_kernel_inner2(double *a, int mm, int nn, int kk, double *a2, int n2, int k2) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= 1 && i <= mm - 2 && j >= 1 && j <= nn - 2 && k >= 1 && k <= kk - 2) {
        a(i, j, k) = (
                             a2(i / 2, j / 2, k / 2) +
                             a2(i / 2, j / 2, k / 2 + k % 2) +
                             a2(i / 2, j / 2 + j % 2, k / 2) +
                             a2(i / 2, j / 2 + j % 2, k / 2 + k % 2) +
                             a2(i / 2 + i % 2, j / 2, k / 2) +
                             a2(i / 2 + i % 2, j / 2, k / 2 + k % 2) +
                             a2(i / 2 + i % 2, j / 2 + j % 2, k / 2) +
                             a2(i / 2 + i % 2, j / 2 + j % 2, k / 2 + k % 2)
                     ) / 8.;
    }
}

void run_jac_kernel_inner2(double *a, int mm, int nn, int kk, double *a2, int n2, int k2) {
    dim3 gridDim = dim3((kk + 31) / 32, (nn + 31) / 32, mm);
    dim3 blockDim = dim3(32, 32, 1);
    jac_kernel_inner2<<<gridDim, blockDim>>>(a, mm, nn, kk, a2, n2, k2);
    CUDA_SAFE_CALL(cudaGetLastError());
}

void barrier() {
    for (int device = 0; device < 2; ++device) {
        CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    CUDA_SAFE_CALL(cudaSetDevice(0));
}

double jac(double *a_gpu, int mm, int nn, int kk, int itmax, double maxeps)
{
    int it;
    double eps;

    if (mm > 31 && nn > 31) {
        int m2 = (mm + 1) / 2, n2 = (nn + 1) / 2, k2 = (kk + 1) / 2;

        double *a2_gpu;
        int vecSizeInner = m2*n2*k2;

        CUDA_SAFE_CALL(cudaMalloc(&a2_gpu, vecSizeInner * sizeof(double)));

        run_jac_kernel_inner1(a_gpu, mm, nn, kk, a2_gpu, n2, k2);

        eps = jac(a2_gpu, m2, n2, k2, itmax * 2, maxeps);

        run_jac_kernel_inner2(a_gpu, mm, nn, kk, a2_gpu, n2, k2);

        CUDA_SAFE_CALL(cudaFree(a2_gpu));
    }

    double *a = (double *) malloc(mm*nn*kk * sizeof(double));
    CUDA_SAFE_CALL(cudaMemcpy(a, a_gpu, mm*nn*kk * sizeof(double), cudaMemcpyDeviceToHost));

    int mid = mm / 2;
    int vecSizes[] = {(mid + 1)*nn*kk, (mm - mid + 1)*nn*kk};
    double **tmp = (double **) malloc(2 * sizeof(double *));
    tmp[0] = (double *) malloc(vecSizes[0] * sizeof(double));
    tmp[1] = (double *) malloc(vecSizes[1] * sizeof(double));

    int i, j, k;
    for (i = 0; i <= mid; i++)
        for (j = 0; j <= nn - 1; j++)
            for (k = 0; k <= kk - 1; k++) {
                tmp(0, i, j, k) = a(i, j, k);
            }

    for (i = mid - 1; i <= mm - 1; i++)
        for (j = 0; j <= nn - 1; j++)
            for (k = 0; k <= kk - 1; k++) {
                tmp(1, i - mid + 1, j, k) = a(i, j, k);
            }

    double **a_2gpus = (double **) malloc(2 * sizeof(double *));
    for (int device = 0; device < 2; ++device) {
        CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
        CUDA_SAFE_CALL(cudaMalloc(&a_2gpus[device], vecSizes[device] * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpyAsync(a_2gpus[device], tmp[device], vecSizes[device] * sizeof(double), cudaMemcpyHostToDevice));
    }
    barrier();

    ///////////////////
    int diffVecSize[] = {(mid - 1)*(nn - 2)*(kk - 2), (mm - mid - 1)*(nn - 2)*(kk - 2)};
    int borderSize = nn * kk;

    double **diff = (double **) malloc(2 * sizeof(double *));
    double **borders = (double **) malloc(2 * sizeof(double *));
    for (int device = 0; device < 2; ++device) {
        CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
        CUDA_SAFE_CALL(cudaMalloc(&diff[device], diffVecSize[device] * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc(&borders[device], borderSize * sizeof(double)));
    }

    double **bordersTmp = (double **) malloc(2 * sizeof(double *));
    bordersTmp[0] = (double *) malloc(borderSize * sizeof(double));
    bordersTmp[1] = (double *) malloc(borderSize * sizeof(double));

    int mms[] = {mid + 1, mm - mid + 1};
    int borderIdxs[] = {mid, 0};
    double epsD[2] = {0.0, 0.0};

    for (it = 1; it <= itmax; it++)
    {
        for (int device = 0; device < 2; ++device) {
            CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
            run_jac_kernel(a_2gpus[device], mms[device], nn, kk, diff[device], borders[device], device == 0);
        }

        for (int device = 0; device < 2; ++device) {
            CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
            epsD[device] = thrust::reduce(
                    thrust::device_pointer_cast<double>(diff[device]),
                    thrust::device_pointer_cast<double>(diff[device]) + diffVecSize[device],
                    0.0f, thrust::maximum<double>()
            );
        }

        for (int device = 0; device < 2; ++device) {
            CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
            CUDA_SAFE_CALL(cudaMemcpyAsync(bordersTmp[device], borders[device], borderSize * sizeof(double), cudaMemcpyDeviceToHost));
        }
        barrier();

        for (int device = 0; device < 2; ++device) {
            CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
            CUDA_SAFE_CALL(cudaMemcpyAsync(borders[device], bordersTmp[1 - device], borderSize * sizeof(double), cudaMemcpyHostToDevice));
            writeBorder(a_2gpus[device], nn, kk, borders[device], borderIdxs[device]);
        }

        eps = Max(epsD[0], epsD[1]);

        if (TRACE && it%TRACE == 0)
            printf("IT=%d eps=%.4g\n", it, eps);
        if (eps < maxeps)
            break;
    }
    for (int device = 0; device < 2; ++device) {
        CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
        CUDA_SAFE_CALL(cudaFree(diff[device]));
        CUDA_SAFE_CALL(cudaFree(borders[device]));
    }
    free(diff);
    free(borders);
    free(bordersTmp[0]);
    free(bordersTmp[1]);
    free(bordersTmp);
    ///////////////////

    for (int device = 0; device < 2; ++device) {
        CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
        CUDA_SAFE_CALL(cudaMemcpyAsync(tmp[device], a_2gpus[device], vecSizes[device] * sizeof(double), cudaMemcpyDeviceToHost));
    }
    barrier();

    for (i = 0; i <= mid - 1; i++)
        for (j = 0; j <= nn - 1; j++)
            for (k = 0; k <= kk - 1; k++) {
                a(i, j, k) = tmp(0, i, j, k);
            }

    for (i = mid; i <= mm - 1; i++)
        for (j = 0; j <= nn - 1; j++)
            for (k = 0; k <= kk - 1; k++) {
                a(i, j, k) = tmp(1, i - mid + 1, j, k);
            }

    CUDA_SAFE_CALL(cudaMemcpy(a_gpu, a, mm*nn*kk * sizeof(double), cudaMemcpyHostToDevice));

    for (int device = 0; device < 2; ++device) {
        free(tmp[device]);
        CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
        CUDA_SAFE_CALL(cudaFree(a_2gpus[device]));
    }
    free(tmp);
    free(a_2gpus);

    barrier();

    return eps;
}


