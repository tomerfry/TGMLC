#include <stdio.h>
#include <malloc.h>


typedef struct {
    size_t rows;
    size_t cols;
} MatrixHeader;

typedef float* Matrix;

float *mat_init(size_t rows, size_t cols) {
    MatrixHeader *mat_header = malloc(sizeof(float) * rows * cols + sizeof(MatrixHeader));
    mat_header->rows = rows;
    mat_header->cols = cols;
    return (Matrix)(mat_header + 1);
}

size_t mat_cols(Matrix mat) {
    return ((MatrixHeader *)mat - 1)->cols;
}

size_t mat_rows(Matrix mat) {
    return ((MatrixHeader *)mat - 1)->rows;
}


int main(int argc, char **argv) {
    float *mat_a = mat_init(4, 4);
    printf("Allocated matrix-A (%ldx%ld)!\n", mat_cols(mat_a), mat_rows(mat_a));
    float *mat_b = mat_init(4, 4);
    printf("Allocated matrix-B (%ldx%ld)!\n", mat_cols(mat_b), mat_rows(mat_b));
    return 0;
}

