#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>


typedef struct {
    size_t rows;
    size_t cols;
} MatrixHeader;

typedef float* Matrix;

typedef struct {
    size_t lcount;
} NeuralNetwork;

typedef struct {
    Matrix *w;
    Matrix *b;
    Matrix *r;
} NNLayer;

typedef NNLayer* NeuralNet;

#define MAT_HEAD(mat) ((MatrixHeader *)(mat) - 1) 
#define COLS(mat) ((MAT_HEAD((mat)))->cols)
#define ROWS(mat) ((MAT_HEAD((mat)))->rows)

#define NN_HEAD(nn) ((NeuralNetwork *)(nn) - 1)
#define COUNT(nn) (NN_HEAD((nn))->lcount)

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

Matrix mat_assign(Matrix mat, float *values) {
    memcpy(mat, values, COLS(mat)*ROWS(mat)*sizeof(float));
    return mat;
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
        for (size_t c = 0; c < cols; ++c) {
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

void mat_add(Matrix a, Matrix b, Matrix result) {
    for (size_t r = 0; r < ROWS(result); ++r) {
        for (size_t c = 0;  c < COLS(result); ++c) {
            *mat_at(result, r, c) = *mat_at(a, r, c) + *mat_at(b, r, c);
        }
    }
}

void mat_dot(Matrix a, Matrix b, Matrix result) {
    for (size_t r = 0; r < ROWS(result); ++r) {
        for (size_t c = 0;  c < COLS(result); ++c) {

            float sum = 0;
            for (size_t i = 0; i < COLS(a); ++i) {
                sum += *mat_at(a, r, i) * *mat_at(b, i, c);
            }

            *mat_at(result, r, c) = sum;
        }
    }
}

NeuralNet nn_init(size_t lcount, size_t input_size, size_t shapes[]) {
    if (lcound < 1 || sizeof(shapes) == 0) return NULL;

    NeuralNetwork *nn_head = malloc(sizeof(NeuralNetwork) + (sizeof(NNLayer) * shapes));
    nn_head->lcount = lcount;
    NeuralNet nn = (NeuralNet)(nn_head + 1);

    nn[0].w = mat_rand(input_size, shapes[0]);
    nn[0].b = mat_rand(1, shapes[0]);
    nn[0].r = mat_rand(1, shapes[0]);

    for (size_t i = 1; i < lcount; ++i) {
        nn[i].w = mat_rand(shapes[i-1], shapes[i]);
        nn[i].b = mat_rand(1, shapes[i]);
        nn[i].r = mat_rand(1, shapes[i]);
    }
    return nn;
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
    float inputs[][2] = {
        {1.0, 1.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {0.0, 0.0}
    };

    float results[] = {
        0.0,
        1.0,
        1.0,
        0.0
    };
    
    srandom(0x1337);
    Matrix x = mat_rand(1, 2);
    Matrix w1 = mat_rand(2, 2);
    Matrix b1 = mat_rand(1, 2);
    Matrix r1 = mat_rand(1, 2);

    Matrix w2 = mat_rand(2, 1);
    Matrix b2 = mat_rand(1, 1);
    Matrix r2 = mat_zeros(1, 1);

    mat_assign(x, inputs[0]);

    //Layer 1
    mat_dot(x, w1, r1);
    mat_add(r1, b1, r1);

    //Layer 2
    mat_dot(r1, w2, r2);
    mat_add(r2, b2, r2);


}

int main(int argc, char **argv) {
    //part_1();
    part_2();
}


