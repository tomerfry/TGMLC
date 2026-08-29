#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>


typedef struct {
    size_t rows;
    size_t cols;
} MatrixHeader;

typedef float* Matrix;


#define MAT_HEAD(mat) ((MatrixHeader *)(mat) - 1) 
#define COLS(mat) ((MAT_HEAD((mat)))->cols)
#define ROWS(mat) ((MAT_HEAD((mat)))->rows)

Matrix mat_zeros(size_t rows, size_t cols) {
    MatrixHeader *mat_header = malloc(sizeof(float) * rows * cols + sizeof(MatrixHeader));
    mat_header->rows = rows;
    mat_header->cols = cols;

    bzero((Matrix)(mat_header + 1), rows*cols*sizeof(float));
    return (Matrix)(mat_header + 1);
}

Matrix mat_init(Matrix bare_matrix, size_t rows, size_t cols) {
    MatrixHeader *mat_header = malloc(sizeof(float) * rows * cols + sizeof(MatrixHeader));
    mat_header->rows = rows;
    mat_header->cols = cols;

    memcpy((Matrix)(mat_header + 1), bare_matrix, rows*cols*sizeof(float));
    return (Matrix)(mat_header + 1);
}

float *mat_at(Matrix mat, size_t r, size_t c) {
    if (r >= MAT_HEAD(mat)->rows || c >= MAT_HEAD(mat)->cols) return NULL;
    return &mat[r * MAT_HEAD(mat)->cols + c];
}

float rand_float() {
    return ((float)random()) / (float)RAND_MAX;
}

Matrix mat_rand(size_t rows, size_t cols) {
    Matrix mat = mat_zeros(rows, cols);
    
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < rows; ++c) {
            *mat_at(mat, r, c) = rand_float();
        }
    }
    return mat;
}

void switch_floats(float *a, float *b) {
    if (*a == *b) return;
    *(int *)a ^= *(int *)b;
    *(int *)b ^= *(int *)a;
    *(int *)a ^= *(int *)b;
}


Matrix mat_transpose(Matrix mat) {
    Matrix temp = mat_zeros(COLS(mat), ROWS(mat));
    
    for (size_t i = 0; i < ROWS(mat); ++i) {
        for (size_t j = 0; j < COLS(mat); ++j) {
            *mat_at(temp, j, i) = *mat_at(mat, i, j);
        }    
    }
    
    free(MAT_HEAD(mat));
    return temp;
}

void mat_print(Matrix mat) {
    size_t rows = MAT_HEAD(mat)->rows;
    size_t cols = MAT_HEAD(mat)->cols;

    for (size_t r = 0;  r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            printf("%lf ", *mat_at(mat, r, c));
        }
        printf("\n");
    }
}

void part_1() {
    float bare_matrix[] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f,
    };
    float *mat_a = mat_zeros(4, 4);
    printf("Allocated matrix-A (%ldx%ld)!\n", MAT_HEAD(mat_a)->rows, MAT_HEAD(mat_a)->cols);
    mat_print(mat_a);
    float *mat_b = mat_init(bare_matrix, 3, 4);
    printf("Allocated matrix-B (%ldx%ld)!\n", MAT_HEAD(mat_b)->rows, MAT_HEAD(mat_b)->cols);
    mat_print(mat_b);
    mat_b = mat_transpose(mat_b);
    printf("Transposed matrix-B (%ldx%ld)!\n", MAT_HEAD(mat_b)->rows, MAT_HEAD(mat_b)->cols);

    mat_print(mat_b);
}

void part_2() {
    float data[] = {
        1.0, 1.0, 0.0,
        1.0, 0.0, 1.0,
        0.0, 1.0, 1.0,
        0.0, 0.0, 0.0
    };
    
    srandom(0x1337);
    Matrix mat = mat_rand(2, 2);
    mat_print(mat);
}

int main(int argc, char **argv) {
    // part_1();
    part_2();
}



