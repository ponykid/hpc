/*
Description: Vector Addition using CPU
Author: Tushar Gautam
Date: 12th Jan 2022
*/

#include <iostream>
#include <chrono>

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
Function that adds two vectors/arrays (h_A and h_B) and
stores them in h_C
Inputs:
    1) *h_A : Pointer to array A
    2) *h_B : Pointer to array A
    3) *h_C : Pointer to array C
    4) N    : Length of each array
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
    for (int i = 0; i < N; i++)
        *(h_C+i) = *(h_A+i) + *(h_B+i);
}
