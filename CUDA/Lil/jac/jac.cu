#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>

#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

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
int TRACE = 1;
double EPS;
int     M, N, K, ITMAX;
double  MAXEPS = 0.1;

double *A, *A_GPU;
double *diff;
thrust::device_ptr<double> diff_dev;

#define A(i,j,k) A[((i)*N+(j))*K+(k)]

double solution(int i, int j, int k)
{
    double x = 10.*i / (M - 1), y = 10.*j / (N - 1), z = 10.*k / (K - 1);
    return 2.*x*x - y*y - z*z;
}

double jac(double *a, int mm, int nn, int kk, int itmax, double maxeps);

int main(int an, char **as)
{
    int i, j, k;

    in = fopen("data3.in", "r");
    if (in == NULL) { printf("Can not open 'data3.in' "); exit(1); }
    i = fscanf(in, "%d %d %d %d %d", &M, &N, &K, &ITMAX, &TRACE);
    if (i < 4) 
    {
        printf("Wrong 'data3.in' (M N K ITMAX TRACE)");
        exit(2);
    }

    A = (double *) malloc(M*N*K * sizeof(double));

    for (i = 0; i <= M - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= K - 1; k++) {
                if (i == 0 || i == M - 1 || j == 0 || j == N - 1 || k == 0 || k == K - 1)
                    A(i, j, k) = solution(i, j, k);
                else 
                    A(i, j, k) = 0.;
            }

    CUDA_SAFE_CALL(cudaMalloc(&A_GPU, M*N*K * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&diff, M*N*K * sizeof(double)));
    diff_dev = thrust::device_pointer_cast<double>(diff);

    CUDA_SAFE_CALL(cudaMemcpy(A_GPU, A, M*N*K * sizeof(double), cudaMemcpyHostToDevice));

    auto time0 = std::chrono::high_resolution_clock::now();

    EPS = jac(A_GPU, M, N, K, ITMAX, MAXEPS);

    auto time1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = time1 - time0;
    printf("%dx%dx%d x %d\t<", M, N, K, ITMAX);
    printf("%3.5f s.>\teps=%.4g\n", elapsed.count(), EPS);

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
    CUDA_SAFE_CALL(cudaFree(diff));
    return 0;
}

#define a(i,j,k) a[((i)*nn+(j))*kk+(k)]
#define b(i,j,k) b[((i)*nn+(j))*kk+(k)]
#define diff(i,j,k) diff[((i)*nn+(j))*kk+(k)]

__global__
void jac_kernel(double *a, int mm, int nn, int kk, double *diff) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= 1 && i <= mm - 2 && j >= 1 && j <= nn - 2 && k >= 1 && k <= kk - 2) {
        double tmp = (a(i - 1, j, k) + a(i + 1, j, k) + a(i, j - 1, k) + a(i, j + 1, k)
                      + a(i, j, k - 1) + a(i, j, k + 1)) / 6.;
        diff(i, j, k) = fabsf(a(i, j, k) - tmp);
        a(i, j, k) = tmp;
    }
}

void run_jac_kernel(double *a, int mm, int nn, int kk, double *diff) {
    dim3 gridDim = dim3((kk + 31) / 32, (nn + 31) / 32, mm);
    dim3 blockDim = dim3(32, 32, 1);
    jac_kernel<<<gridDim, blockDim>>>(a, mm, nn, kk, diff);
    CUDA_SAFE_CALL(cudaGetLastError());
}

double jac(double *a_gpu, int mm, int nn, int kk, int itmax, double maxeps)
{
    int it, vecSize = mm*nn*kk;
    double eps;

    for (it = 1; it <= itmax - 1; it++)
    {
        run_jac_kernel(a_gpu, mm, nn, kk, diff);
        eps = thrust::reduce(
                diff_dev, diff_dev + vecSize, 0.0f, thrust::maximum<double>()
        );

        if (TRACE && it%TRACE == 0)
            printf("\nIT=%d eps=%.4g\t", it, eps);
        if (eps < maxeps) 
            break;
    }
    return eps;
}
