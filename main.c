#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>
#include <math.h>

// gcc -g main.c -lm

typedef struct {
    size_t rows;
    size_t cols;
} MatrixHeader;

typedef float* Matrix;

typedef struct {
    size_t lcount;
} NeuralNetwork;

typedef struct {
    Matrix input;
    Matrix w;
    Matrix b;
    Matrix result;
} NNLayer;

typedef NNLayer* NeuralNet;

#define MAT_HEAD(mat) ((MatrixHeader *)(mat) - 1) 
#define COLS(mat) ((MAT_HEAD((mat)))->cols)
#define ROWS(mat) ((MAT_HEAD((mat)))->rows)

#define NN_HEAD(nn) ((NeuralNetwork *)(nn) - 1)
#define NN_COUNT(nn) (NN_HEAD((nn))->lcount)
#define NN_LAST_LAYER(nn) ((nn)[NN_COUNT((nn))-1])
#define EPS (0.01f)

float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

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

void mat_apply(Matrix mat, float (*activation)(float)) {
    for (size_t r = 0; r < ROWS(mat); ++r) {
        for (size_t c = 0;  c < COLS(mat); ++c) {
            *mat_at(mat, r, c) = (*activation)(*mat_at(mat, r, c));
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
    if (lcount < 1) return NULL;

    NeuralNetwork *nn_head = malloc(sizeof(NeuralNetwork) + (sizeof(NNLayer) * lcount));
    nn_head->lcount = lcount;
    NeuralNet nn = (NeuralNet)(nn_head + 1);

    nn[0].input = mat_zeros(1, input_size);
    nn[0].w = mat_rand(input_size, shapes[0]);
    nn[0].b = mat_rand(1, shapes[0]);
    nn[0].result = mat_rand(1, shapes[0]);

    for (size_t i = 1; i < lcount; ++i) {
        nn[i].input = mat_zeros(1, shapes[i-1]);
        nn[i].w = mat_rand(shapes[i-1], shapes[i]);
        nn[i].b = mat_rand(1, shapes[i]);
        nn[i].result = mat_rand(1, shapes[i]);
    }
    return nn;
}

void nn_passthrough(NeuralNet nn, float input[]) {
    
    mat_assign(nn[0].input, input);
    mat_dot(nn[0].input, nn[0].w, nn[0].result);
    mat_add(nn[0].result, nn[0].b, nn[0].result);
    mat_apply(nn[0].result, &sigmoidf);

    for (size_t l = 1; l < NN_COUNT(nn); ++l) {
        mat_assign(nn[l].input, nn[l-1].result);
        mat_dot(nn[l].input, nn[l].w, nn[l].result);
        mat_add(nn[l].result, nn[l].b, nn[l].result);
        mat_apply(nn[l].result, &sigmoidf);
    }
}

int main(int argc, char **argv) {
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
    size_t shapes[] = {2, 2};
    NeuralNet nn = nn_init(2, 2, shapes);
    
    for (int i = 0; i < 4; ++i) {
        nn_passthrough(nn, inputs[i]);
        printf("%lf\n", *mat_at(NN_LAST_LAYER(nn).result, 0, 0));
    }
} 
