/*
Description: GPU Matrix Matrix multiplication (square matrix)
Author: Tushar Gautam
Date: 16th Jan 2022
*/

#include <cuda.h>
#include <iostream>

using namespace std;

// Functions
void print(float *mat, int N);
/*
Function to print Array
Inputs:
    1) **arr : Pointer of Pointer to 2D array
    2) N : Size of square matrix
*/

void matMul(float *A, float *B, float *C, int N);
/*
Function to perform matrix matrix multiplication of
A and B on the CPU, and stores in C.
Inputs:
    1) **A : Matrix A
    2) **B : Matrix B
    3) **C : Matrix C
    4) N : Size of square matrix
*/

__global__ void MatMulKernel(float *d_A, float *d_B, float *d_C, int N);
/*
GPU kernel to perform matrix matrix multiplication of
A and B, and stores in C.
Inputs:
    1) **d_A : Matrix A
    2) **d_B : Matrix B
    3) **d_C : Matrix C
    4) N : Size of square matrix
*/

void checkError(cudaError_t err);
/*
Function that checks memory allocation error
Inputs:
    1) err : error collected from cudaMalloc
*/

int main() {
    int N; // Size of square matrix
    std::cout << "Enter size of square matrix: ";
    std::cin >> N;

    // Allocating matrices dynamically
    float *A, *B, *C;
    A = new float [N*N];
    B = new float [N*N];
    C = new float [N*N];

    // Initialising matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i*N + j] = (float)(i+1)*(j+1);
            B[i*N + j] = (float)(i+1)/(j+1);
            C[i*N + j] = 0.0;
        }
    }

    // Printing
    std::cout << "Matrix A: \n";
    print(A, N);

    std::cout << "Matrix B: \n";
    print(B, N);

    // Mat Mul
    matMul(A, B, C, N);

    std::cout << "Matrix C: \n";
    print(C, N);

    return 0;
}

void print(float *mat, int N) {
    std::cout << "Matrix: \n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << mat[i*N + j] << " ";
        std::cout << "\n";
    }
}

void matMul(float *A, float *B, float *C, int N) {
    int size = N * N * sizeof(float);

    // Pointers to matrices in device memory
    float *d_A, *d_B, *d_C;

    // Allocating and copying matrices
    cudaError_t errA = cudaMalloc(&d_A, size);
    checkError(errA);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    cudaError_t errB = cudaMalloc(&d_B, size);
    checkError(errB);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaError_t errC = cudaMalloc(&d_C, size);
    checkError(errC);
    //cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    // Kernel execution
    int numThrds = 2;
    // Grid Parameters
    dim3 gridDim(ceil(N/(float)numThrds), ceil(N/(float)numThrds));
    dim3 blockDim(numThrds, numThrds);
    // Calling kernel
    MatMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copying back results
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Freeing memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


__global__ void MatMulKernel(float *d_A, float *d_B, float *d_C, int N) {
    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if ((rowIdx < N) && (colIdx < N)) {
        float sum = 0;
        for (int k = 0; k < N; k++)
            sum += d_A[rowIdx*N+k] * d_B[k*N+colIdx];
        d_C[rowIdx*N+colIdx] = sum;
    }
}


void checkError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__
                    << " at line " << __LINE__ << "\n";
        exit(EXIT_FAILURE);
    }
}
