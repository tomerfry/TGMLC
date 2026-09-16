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
    Matrix w_grad;
    Matrix b_grad;
    Matrix temp_grad;
} NNLayer;

typedef NNLayer* NeuralNet;

#define MAT_HEAD(mat) ((MatrixHeader *)(mat) - 1) 
#define COLS(mat) ((MAT_HEAD((mat)))->cols)
#define ROWS(mat) ((MAT_HEAD((mat)))->rows)

#define NN_HEAD(nn) ((NeuralNetwork *)(nn) - 1)
#define NN_COUNT(nn) (NN_HEAD((nn))->lcount)
#define NN_LAST_LAYER(nn) ((nn)[NN_COUNT((nn))-1])

#define EPS (1e-3)
#define RATE (1.0)
#define BITS 4
#define N_SAMPLES (1 << (2*BITS))      // 256


float rand_float();
float sigmoidf(float x);
void switch_floats(float *a, float *b);

Matrix mat_zeros(size_t rows, size_t cols);
Matrix mat_init(Matrix bare_matrix, size_t rows, size_t cols); 
Matrix mat_assign(Matrix mat, float *values);
float *mat_at(Matrix mat, size_t r, size_t c);
Matrix mat_rand(size_t rows, size_t cols);
Matrix mat_transpose(Matrix mat);
void mat_print(Matrix mat);
void mat_add(Matrix a, Matrix b, Matrix result);
void mat_apply(Matrix mat, float (*activation)(float));
void mat_dot(Matrix a, Matrix b, Matrix result);

NeuralNet nn_init(size_t inputs_amount, size_t lcount, size_t shapes[]);
void nn_passthrough(NeuralNet nn, float input[]);
float nn_cost(NeuralNet nn, Matrix inputs, Matrix results);
void nn_finite_diff(NeuralNet nn, Matrix inputs, Matrix results);
void nn_backprop(NeuralNet nn, Matrix inputs, Matrix results);
void nn_learn(NeuralNet nn, float inputs[], float results[]);
void nn_print(NeuralNet nn);
void nn_train(NeuralNet nn, Matrix inputs, Matrix results);
