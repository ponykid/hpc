/*
Description: Vector Addition using CPU
Author: Tushar Gautam
Date: 12th Jan 2022
*/

#include <iostream>
#include <chrono>
#include <cuda.h>

using namespace std;
using namespace std::chrono;

// Functions
void print(float *arr, int N);
/*
Function to print Array
Inputs:
    1) *arr : Pointer to array
    2) N    : Length of array
*/

void vecAdd(float *h_A, float *h_B, float *h_C, int N);
/*
Host code that calls device Kernel (vecAddKernel) which adds
vector h_A and h_B on a GPU and stores in h_C
Inputs:
    1) *h_A : Pointer to array A
    2) *h_B : Pointer to array B
    3) *h_C : Pointer to array C
    4) N    : Length of each array
*/

__global__ void vecAddKernel(float *A, float *B, float *C, int N);
/*
Device kernel code that adds vectors A and B on a GPU and stores in C
Inputs:
    1) *A : Pointer to array A in GPU memory
    2) *B : Pointer to array B in GPU memory
    3) *C : Pointer to array C in GPU memory
    4) N    : Length of each array
*/

void checkError(cudaError_t err);
/*
Function that checks memory allocation error
Inputs:
    1) err : error collected from cudaMalloc
*/


int main() {
    int N = 111900;

    float A[N], B[N], C[N];
    float *h_A = 0;
    float *h_B = 0;
    float *h_C = 0;

    h_A = A;
    h_B = B;
    h_C = C;

    // Initialise Arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i+2;
        h_B[i] = i+3;
        h_C[i] = 0;
    }

    //std::cout << "Printing array A: \n"; print(h_A, N);
    //std::cout << "Printing array B: \n"; print(h_B, N);
    auto start = high_resolution_clock::now();
    vecAdd(h_A, h_B, h_C, N);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);

    std::cout << "Execution time for Array of size "<< N << ": " << (double)duration.count()/1000
                << " Microseconds" << '\n';

    //std::cout << "Printing array C: \n"; print(h_C, N);
    return 0;
}


void print(float *arr, int N) {
    std::cout << "Array: ";
    for (int i = 0; i < N; i++)
        std::cout << arr[i] << " ";
    std::cout << "\n";
}

void vecAdd(float *h_A, float *h_B, float *h_C, int N) {
    // Calculating size in bytes
    int size = N * sizeof(float);
    // Pointers to arrays in device memory
    float *d_A, *d_B, *d_C;

    // Allocating and copying A
    cudaError_t errA = cudaMalloc(&d_A, size);
    checkError(errA);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Allocating and copying B
    cudaError_t errB = cudaMalloc(&d_B, size);
    checkError(errB);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Allocating C
    cudaError_t errC = cudaMalloc(&d_C, size);
    checkError(errC);

    // Kernel Invocation
    dim3 dimGrid(ceil(N/256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    vecAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Copying back result
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void vecAddKernel(float *A, float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void checkError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__
                    << " at line " << __LINE__ << "\n";
        exit(EXIT_FAILURE);
    }
}
