/*
Description:    Scale each value of a 2D array by 2
Author: Tushar Gautam
Date: 13th Jan 2022
*/

#include <iostream>

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
Function to scale Array
Inputs:
    1) *arr : Pointer to 2D array
    2) row  : Number of rows
    2) col  : Number of cols
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
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            pic[i*col+j] *= 2;
    }
}
