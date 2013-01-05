#define SIZE 16

const int A[SIZE][SIZE] = {
#include "inputs/matrixA.txt"
};
const int B[SIZE][SIZE] = {
#include "inputs/matrixB.txt"
};
int C[SIZE][SIZE] = { 0 };

int main() {
    write_pair(1, "threadIdx.y: ", threadIdx.y);
    write_pair(1, "threadIdx.x: ", threadIdx.x);
    write_pair(1, "threadDim.y: ", threadDim.y);
    write_pair(1, "threadDim.x: ", threadDim.x);
    write_pair(1, "blockIdx.y: ", blockIdx.y);
    write_pair(1, "blockIdx.x: ", blockIdx.x);
    write_pair(1, "blockDim.y: ", blockDim.y);
    write_pair(1, "blockDim.x: ", blockDim.x);

    int i;
    int sum = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (i = 0; i < SIZE; ++i) {
        sum += A[row][i] * B[i][col];
    }
    C[row][col] = sum;

    write_pair(1, "result.row= ", row);
    write_pair(1, "result.col= ", col);
    write_pair(1, "result= ", sum);
    return 0;
}
