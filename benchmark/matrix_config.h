#ifndef MATRIX_CONFIG_H
#define MATRIX_CONFIG_H

#ifndef SIZE
#error "Please specify the size for the matrix program"
#endif

#if SIZE == 16
#define MATRIX_A_INPUT_FILE "inputs/matrix16_A.txt"
#define MATRIX_B_INPUT_FILE "inputs/matrix16_B.txt"
#elif SIZE == 256
#define MATRIX_A_INPUT_FILE "inputs/matrix256_A.txt"
#define MATRIX_B_INPUT_FILE "inputs/matrix256_B.txt"
#endif

#endif /* MATRIX_CONFIG_H */
