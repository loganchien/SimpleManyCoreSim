#include "matrix_config.h"

const int A[SIZE][SIZE] = {
#include MATRIX_A_INPUT_FILE
};

const int B[SIZE][SIZE] = {
#include MATRIX_B_INPUT_FILE
};

int C[SIZE][SIZE] = { 0 };

int main() {
    int i;
    int sum = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (i = 0; i < SIZE; ++i) {
        sum += A[row][i] * B[i][col];
    }
    C[row][col] = sum;

    //write_pair(1, "result.row= ", row);
    //write_pair(1, "result.col= ", col);
    //dump1(1, sum);
    return 0;
}
