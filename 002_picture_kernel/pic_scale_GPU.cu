/*
Description:    Scale each value of a 2D array
                by 2 using GPU
Author: Tushar Gautam
Date: 13th Jan 2022
*/

#include <iostream>
#include <cuda.h>

using namespace std;

// Functions
void print(float *pic, int row, int col);
/*
Function to print Array
Inputs:
    1) *arr : Pointer to 2D array
    2) row  : Number of rows
    2) col  : Number of cols
*/

void scalePic(float *pic, int row, int col);
/*
Host function that calls GPU kernel which scales Array
Inputs:
    1) *arr : Pointer to 2D array
    2) row  : Number of rows
    2) col  : Number of cols
*/

__global__ void scalePicKernel(float *d_pic, int row, int col);
/*
Device kernel that scales Array
Inputs:
    1) *d_pic : Pointer to 2D array in GPU memory
    2) row  : Number of rows
    2) col  : Number of cols
*/

void checkError(cudaError_t err);
/*
Function that checks memory allocation error
Inputs:
    1) err : error collected from cudaMalloc
*/


int main() {
    int row = 5;
    int col = 4;

    float pic[row*col];
    float *ptr = 0;

    ptr = pic;

    // Initialising picture
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            ptr[i*col+j] = i+j;
    }

    // Printing pic
    std::cout << "Original Matrix: \n";
    print(ptr, row, col);

    // Scaling picture
    scalePic(ptr, row, col);

    // Printing pic
    std::cout << "Transformed Matrix: \n";
    print(ptr, row, col);

    return 0;

}

void print(float *pic, int row, int col) {
    std::cout << "Matrix: \n";
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            std::cout << pic[i*col+j] << " ";
        std::cout << "\n";
    }
}

void scalePic(float *pic, int row, int col) {
    // Calculating size to be allocated
    int size = row * col * sizeof(float);

    // Pointer to array in device memory
    float *d_pic = 0;

    // Allocating and copying pic to device memory
    cudaError_t err = cudaMalloc(&d_pic, size);
    checkError(err);
    cudaMemcpy(d_pic, pic, size, cudaMemcpyHostToDevice);

    // Kernel invocation
    int numThrds = 16;
    // Defining grid parameters
    dim3 gridDim(ceil(col/(float)numThrds), ceil(row/(float)numThrds), 1);
    dim3 blockDim(numThrds, numThrds, 1);
    // Kernel execution
    scalePicKernel<<<gridDim, blockDim>>>(d_pic, row, col);

    // Copying back results and freeing memory
    cudaMemcpy(pic, d_pic, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pic);
}

__global__ void scalePicKernel(float *d_pic, int row, int col) {
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;

    if ((rowIdx < row) && (colIdx < col))
        d_pic[rowIdx*col+colIdx] *= 2;
}

void checkError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__
                    << " at line " << __LINE__ << "\n";
        exit(EXIT_FAILURE);
    }
}
