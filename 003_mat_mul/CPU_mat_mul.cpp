/*
Description: Matrix Matrix multiplication (square matrix)
Author: Tushar Gautam
Date: 16th Jan 2022
*/

#include <cuda.h>
#include <iostream>

using namespace std;

// FUnctions
void print(float **mat, int N);
/*
Function to print Array
Inputs:
    1) **arr : Pointer of Pointer to 2D array
    2) N : Size of square matrix
*/

void matMul(float **A, float **B, float **C, int N);
/*
Function to perform matrix matrix multiplication of
A and B, stores in C.
Inputs:
    1) **A : Matrix A
    2) **B : Matrix B
    3) **C : Matrix C
    4) N : Size of square matrix
*/


int main() {
    int N; // Size of square matrix
    std::cout << "Enter size of square matrix: ";
    std::cin >> N;

    // Allocating matrices dynamically
    float **A, **B, **C;
    A = new float *[N];
    B = new float *[N];
    C = new float *[N];

    for (int i = 0; i < N; i++) {
        A[i] = new float [N];
        B[i] = new float [N];
        C[i] = new float [N];
    }

    // Initialising matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)(i+1)*(j+1);
            B[i][j] = (float)(i+1)/(j+1);
            C[i][j] = 0.0;
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

void print(float **mat, int N) {
    std::cout << "Matrix: \n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << mat[i][j] << " ";
        std::cout << "\n";
    }
}

void matMul(float **A, float **B, float **C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
