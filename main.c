#include <stdio.h>
#include <malloc.h>


typedef struct {
    size_t rows;
    size_t cols;
} MatrixHeader;


float *mat_init(size_t rows, size_t cols) {
    MatrixHeader *mat_header = malloc(sizeof(float) * rows * cols + sizeof(MatrixHeader));
    mat_header->rows = rows;
    mat_header->cols = cols;
    return (float *)(mat_header + 1);
}

int main(int argc, char **argv) {
    float *mat = mat_init(4, 4);
    printf("Allocated matrix!\n");
    return 0;
}

