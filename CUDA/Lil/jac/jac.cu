#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/swap.h>

#define  Max(a,b) ((a)>(b)?(a):(b))

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
int     M, N, K, MID, ITMAX;
double  MAXEPS = 0.1;
int devCount = 0;

double **A, **A_GPU;

#define A(d, i, j, k) A[d][((i)*N+(j))*K+(k)]

double solution(int i, int j, int k)
{
    double x = 10.*i / (M - 1), y = 10.*j / (N - 1), z = 10.*k / (K - 1);
    return 2.*x*x - y*y - z*z;
}

double jac(double **a, int mm, int nn, int kk, int mid, int itmax, double maxeps);

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

    MID = M / 2;
    A = (double **) malloc(2 * sizeof(double *));
    A[0] = (double *) malloc((MID + 1)*N*K * sizeof(double));
    A[1] = (double *) malloc((M - MID + 1)*N*K * sizeof(double));

    for (i = 0; i <= MID; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= K - 1; k++) {
                A(0, i, j, k) = (i == 0 || i == M - 1 || j == 0 || j == N - 1 || k == 0 || k == K - 1) ?
                              solution(i, j, k) : 0.;
            }

    for (i = MID - 1; i <= M - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= K - 1; k++) {
                A(1, i - MID + 1, j, k) = (i == 0 || i == M - 1 || j == 0 || j == N - 1 || k == 0 || k == K - 1) ?
                              solution(i, j, k) : 0.;
            }

    int vecSize[] = {(MID + 1)*N*K, (M - MID + 1)*N*K};
    A_GPU = (double **) malloc(2 * sizeof(double *));
    for (int device = 0; device < 2; ++device) {
        CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
        CUDA_SAFE_CALL(cudaMalloc(&A_GPU[device], vecSize[device] * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpy(A_GPU[device], A[device], vecSize[device] * sizeof(double), cudaMemcpyHostToDevice));
    }

    clock_t t = clock();

    EPS = jac(A_GPU, M, N, K, MID, ITMAX, MAXEPS);

    t = clock() - t;
    double elapsed = 1.0 * t / CLOCKS_PER_SEC;
    printf("%dx%dx%d x %d\t<", M, N, K, ITMAX);
    printf("%3.5f s.>\teps=%.4g\n", elapsed, EPS);

    for (int device = 0; device < 2; ++device) {
        CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
        CUDA_SAFE_CALL(cudaMemcpy(A[device], A_GPU[device], vecSize[device] * sizeof(double), cudaMemcpyDeviceToHost));
    }

    if (TRACE)
    {
        EPS = 0.;
        for (i = 0; i <= MID; i++)
            for (j = 0; j <= N - 1; j++)
                for (k = 0; k <= K - 1; k++)
                    EPS = Max(fabs(A(0, i, j, k) - solution(i, j, k)), EPS);

        for (i = MID - 1; i <= M - 1; i++)
            for (j = 0; j <= N - 1; j++)
                for (k = 0; k <= K - 1; k++)
                    EPS = Max(fabs(A(1, i - MID + 1, j, k) - solution(i, j, k)), EPS);
        printf("delta=%.4g\n", EPS);
    }

    for (int device = 0; device < 2; ++device) {
        free(A[device]);
        CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
        CUDA_SAFE_CALL(cudaFree(A_GPU[device]));
    }
    free(A);
    free(A_GPU);

    return 0;
}

#define a(i,j,k) a[((i)*nn+(j))*kk+(k)]
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
    dim3 gridDim = dim3((kk + 15) / 16, (nn + 15) / 16, mm);
    dim3 blockDim = dim3(16, 16, 1);
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
    dim3 gridDim = dim3((kk + 15) / 16, (nn + 15) / 16);
    dim3 blockDim = dim3(16, 16);
    writeBorder_kernel<<<gridDim, blockDim>>>(a, nn, kk, border, borderIdx);
    CUDA_SAFE_CALL(cudaGetLastError());
}

void barrier() {
    for (int device = 0; device < 2; ++device) {
        CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
}

double jac(double **a_gpu, int mm, int nn, int kk, int mid, int itmax, double maxeps)
{
    int it;
    double eps;
    int vecSize[] = {(mid - 1)*(nn - 2)*(kk - 2), (mm - mid - 1)*(nn - 2)*(kk - 2)};
    int borderSize = nn * kk;

    double **diff = (double **) malloc(2 * sizeof(double *));
    double **borders = (double **) malloc(2 * sizeof(double *));
    for (int device = 0; device < 2; ++device) {
        CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
        CUDA_SAFE_CALL(cudaMalloc(&diff[device], vecSize[device] * sizeof(double)));
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
            run_jac_kernel(a_gpu[device], mms[device], nn, kk, diff[device], borders[device], device == 0);
        }

        for (int device = 0; device < 2; ++device) {
            CUDA_SAFE_CALL(cudaSetDevice(device % devCount));
            epsD[device] = thrust::reduce(
                    thrust::device_pointer_cast<double>(diff[device]),
                    thrust::device_pointer_cast<double>(diff[device]) + vecSize[device],
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
            writeBorder(a_gpu[device], nn, kk, borders[device], borderIdxs[device]);
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
    return eps;
}
