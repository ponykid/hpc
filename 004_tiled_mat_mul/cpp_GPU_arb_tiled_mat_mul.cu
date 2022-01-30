/*
Description: Tiled MatMul for arbitrary sized matrices
Author: Tushar Gautam
Date: 29 Jan 2022
*/

#include <cuda.h>
#include <iostream>
#include <assert.h>
#include <math.h>

using namespace std;

#define TILE_SIZE 16

// Functions
void print(double *mat, int N, int M);
/*
Function to print Array
Inputs:
    1) **arr : Pointer of Pointer to 2D array
    2) N : Rows of square matrix
    2) M : Cols of square matrix
*/

void matMul(double *A, double *B, double *C, int n, int k, int m);
/*
Function to perform matrix matrix multiplication of
A and B on the CPU, and stores in C.
Inputs:
    1) *A : Matrix A
    2) *B : Matrix B
    3) *C : Matrix C
    4) n,k,m : Size of square matrix
*/

__global__ void matMulKernel(double *A, double *B, double *C, int n, int k, int m);
/*
GPU kernel to perform matrix matrix multiplication of
A and B, and stores in C.
Inputs:
    1) **d_A : Matrix A
    2) **d_B : Matrix B
    3) **d_C : Matrix C
    4) n,k,m : Size of square matrix
*/

void checkError(cudaError_t err);
/*
Function that checks memory allocation error
Inputs:
    1) err : error collected from cudaMalloc
*/

int main() {
    // Device querying
    int numDev;
    cudaGetDeviceCount(&numDev);

    cudaDeviceProp devProp;
    for (int i = 0; i < numDev; i++){
        cudaGetDeviceProperties(&devProp, i);
        std::cout << "\nDevice Name: " << devProp.name << "\n";
        std::cout << "Number of Streaming processors in device: " << devProp.multiProcessorCount << "\n";
        std::cout << "Max Threads per block: " << devProp.maxThreadsPerBlock << "\n";
        std::cout << "Shared Memory per block: " << devProp.sharedMemPerBlock/1000.0 << " KB \n";

        
    }

    int n, m, k;

    std::cout << "Enter dimensions of matrix A (n,k): ";
    std::cin >> n >> k;

    std::cout << "Enter dimensions of matrix B (k,m): ";
    std::cin >> k >> m;

    double *A;
    A = new double [n*k];

    double *B;
    B = new double [k*m];

    double *C;
    C = new double [n*m];

    // Initialising matrices
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            A[i*k+j] = (double)(i+1)+(j+1);
    
    for (int i = 0; i < k; i++)
        for (int j = 0; j < m; j++)
            B[i*m+j] = (double)(i+1)/(j+1);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[i*m+j] = (double)(0);


    // Matmul on GPU
    matMul(A, B, C, n, k, m);

    // Asserting results
    double *Cref;
    Cref = new double [n*m];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            Cref[i*m+j] = (double)(0);

    // CPU matmul
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < m; j++) 
            for (int z = 0; z < k; z++) 
                Cref[i*m+j] += A[i*k+z] * B[z*m+j];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) 
            if (abs(C[i] - Cref[i]) > pow(10, -5)) {
                printf("Error: mismatch at linearized index %d, was: %f, should be: %f\n", i, C[i], Cref[i]); return -1;
            }
  return 0;
}

void print(double *mat, int N, int M) {
    std::cout << "Matrix: \n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++)
            std::cout << mat[i*M + j] << " ";
        std::cout << "\n";
    }
}

void matMul(double *A, double *B, double *C, int n, int k, int m) {
    // Allocating global memory on GPU and copying data
    double *d_A, *d_B, *d_C;

    cudaError_t errA = cudaMalloc(&d_A, n*k*sizeof(double));
    checkError(errA);
    cudaMemcpy(d_A, A, n*k*sizeof(double), cudaMemcpyHostToDevice);

    cudaError_t errB = cudaMalloc(&d_B, k*m*sizeof(double));
    checkError(errA);
    cudaMemcpy(d_B, B, k*m*sizeof(double), cudaMemcpyHostToDevice);

    cudaError_t errC = cudaMalloc(&d_C, n*m*sizeof(double));
    checkError(errC);
    cudaMemcpy(d_C, C, n*m*sizeof(double), cudaMemcpyHostToDevice);

    // Kernel invocation
    dim3 gridDim(ceil((double)m / TILE_SIZE), ceil((double)n / TILE_SIZE), 1);
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);

    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n, k, m);

    // Copying back results
    cudaMemcpy(C, d_C, n*m*sizeof(double), cudaMemcpyDeviceToHost);

    // Freeing memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void matMulKernel(double *A, double *B, double *C, int n, int k, int m) {
    // Allocating shared memory for tiles
    __shared__ double ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ double ds_B[TILE_SIZE][TILE_SIZE];

    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;

    double cVal = 0;

    for (int t = 0; t < k; t+=TILE_SIZE) {
        // Loading elements into shared memory
        if (rowIdx < n && (t+threadIdx.x) < k)
            ds_A[threadIdx.y][threadIdx.x] = A[rowIdx*k+t+threadIdx.x];
        else
            ds_A[threadIdx.y][threadIdx.x] = 0.0;

        if ((t+threadIdx.y) < k && colIdx < m)
            ds_B[threadIdx.y][threadIdx.x] = B[(t+threadIdx.y)*m+colIdx];
        else
            ds_B[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();

        // Performing computations
        for (int i = 0; i < TILE_SIZE; i++)
            cVal += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        __syncthreads();
    }
    C[rowIdx*m + colIdx] = cVal;
}

void checkError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__
                    << " at line " << __LINE__ << "\n";
        exit(EXIT_FAILURE);
    }
}