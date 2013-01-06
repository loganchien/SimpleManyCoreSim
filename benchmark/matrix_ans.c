#include <stdio.h>
#include <stdlib.h>

#include "matrix_config.h"

const int A[SIZE][SIZE] = {
#include MATRIX_A_INPUT_FILE
};
const int B[SIZE][SIZE] = {
#include MATRIX_B_INPUT_FILE
};

int main() {
    int i, j, k;

    for (i = 0; i < SIZE; ++i) {
        for (j = 0; j < SIZE; ++j) {
            int sum = 0;
            for (k = 0; k < SIZE; ++k) {
                sum += A[i][k] * B[k][j];
            }
            printf("[threadIdx: %08x %08x , blockIdx: %08x %08x] %08x\n",
                   i, j, 0, 0, sum);
        }
    }
    return 0;
}
