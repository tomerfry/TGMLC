#include <stdio.h>
#include <string.h>
#include <malloc.h>


typedef struct {
    size_t rows;
    size_t cols;
} MatrixHeader;

typedef float* Matrix;

float *mat_zeros(size_t rows, size_t cols) {
    MatrixHeader *mat_header = malloc(sizeof(float) * rows * cols + sizeof(MatrixHeader));
    mat_header->rows = rows;
    mat_header->cols = cols;
    return (Matrix)(mat_header + 1);
}

float *mat_init(Matrix bare_matrix, size_t rows, size_t cols) {
    MatrixHeader *mat_header = malloc(sizeof(float) * rows * cols + sizeof(MatrixHeader));
    mat_header->rows = rows;
    mat_header->cols = cols;

    memcpy((Matrix)(mat_header + 1), bare_matrix, rows*cols*sizeof(float));

    return (Matrix)(mat_header + 1);
}

size_t mat_cols(Matrix mat) {
    return ((MatrixHeader *)mat - 1)->cols;
}

size_t mat_rows(Matrix mat) {
    return ((MatrixHeader *)mat - 1)->rows;
}

void mat_transpose(Matrix mat) {
    

    ((MatrixHeader)mat - 1)->rows ^= ((MatrixHeader)mat - 1)->cols;
    ((MatrixHeader)mat - 1)->cols ^= ((MatrixHeader)mat - 1)->rows;
    ((MatrixHeader)mat - 1)->rows ^= ((MatrixHeader)mat - 1)->cols;
};

int mat_at(Matrix mat, size_t r, size_t c, float *cell) {
    if (r >= mat_rows(mat) || c >= mat_cols(mat)) return -1;

    *cell = mat[r * mat_cols(mat) + c];
    return 0;
}


void mat_print(Matrix mat) {
    float cell = 0.0f;
    size_t rows = mat_rows(mat);
    size_t cols = mat_cols(mat);

    for (size_t r = 0;  r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            (void)mat_at(mat, r, c, &cell);
            printf("%lf ", cell);
        }
        printf("\n");
    }
}


int main(int argc, char **argv) {
    float bare_matrix[] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.2f, 0.3f, 0.4f, 0.1f,
        0.3f, 0.4f, 0.1f, 0.2f,
        0.4f, 0.1f, 0.2f, 0.3f
    };
    float *mat_a = mat_zeros(4, 4);
    printf("Allocated matrix-A (%ldx%ld)!\n", mat_rows(mat_a), mat_cols(mat_a));
    mat_print(mat_a);
    float *mat_b = mat_init(bare_matrix, 4, 4);
    printf("Allocated matrix-B (%ldx%ld)!\n", mat_rows(mat_b), mat_cols(mat_b));
    mat_print(mat_b);
    return 0;
}

