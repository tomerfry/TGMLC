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
#define RATE (1e-1)

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

NeuralNet nn_init(size_t inputs_amount, size_t lcount, size_t shapes[]) {
    if (lcount < 1) return NULL;

    NeuralNetwork *nn_head = malloc(sizeof(NeuralNetwork) + (sizeof(NNLayer) * lcount));
    nn_head->lcount = lcount;
    NeuralNet nn = (NeuralNet)(nn_head + 1);

    nn[0].input = mat_zeros(1, inputs_amount);
    nn[0].w = mat_rand(inputs_amount, shapes[0]);
    nn[0].b = mat_rand(1, shapes[0]);
    nn[0].result = mat_zeros(1, shapes[0]);
    nn[0].w_grad = mat_zeros(inputs_amount, shapes[0]);
    nn[0].b_grad = mat_zeros(1, shapes[0]);
    nn[0].temp_grad = mat_zeros(1, shapes[0]);

    for (size_t i = 1; i < lcount; ++i) {
        nn[i].input = mat_zeros(1, shapes[i-1]);
        nn[i].w = mat_rand(shapes[i-1], shapes[i]);
        nn[i].b = mat_rand(1, shapes[i]);
        nn[i].result = mat_zeros(1, shapes[i]);
        nn[i].w_grad = mat_zeros(shapes[i-1], shapes[i]);
        nn[i].b_grad = mat_zeros(1, shapes[i]);
        nn[i].temp_grad = mat_zeros(1, shapes[i]);
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

float nn_cost(NeuralNet nn, Matrix inputs, Matrix results) {
    float sum = 0.0f;
    for (size_t r = 0; r < ROWS(inputs); ++r) {
        nn_passthrough(nn, mat_at(inputs, r, 0));
        float v = *mat_at(results, r, 0) - *mat_at(NN_LAST_LAYER(nn).result, 0, 0);
        sum += v*v;
    }

    return sum/ROWS(inputs);
}

void nn_finite_diff(NeuralNet nn, Matrix inputs, Matrix results) {
    float cost = nn_cost(nn, inputs, results);
    float saved;

    for (size_t l = 0; l < NN_COUNT(nn); ++l) {
        for (size_t r = 0; r < ROWS(nn[l].w); ++r) {
            for (size_t c = 0; c < COLS(nn[l].w); ++c) {
                saved = *mat_at(nn[l].w, r, c);
                *mat_at(nn[l].w, r, c) += EPS;
                *mat_at(nn[l].w_grad, r, c) = (nn_cost(nn, inputs, results) - cost) / EPS;
                *mat_at(nn[l].w, r, c) = saved;
            }
        }

        for (size_t r = 0; r < ROWS(nn[l].b); ++r) {
            for (size_t c = 0; c < COLS(nn[l].b); ++c) {
                saved = *mat_at(nn[l].b, r, c);
                *mat_at(nn[l].b, r, c) += EPS;
                *mat_at(nn[l].b_grad, r, c) = (nn_cost(nn, inputs, results) - cost) / EPS;
                *mat_at(nn[l].b, r, c) = saved;
            }               
        }

    }
}

void nn_backprop(NeuralNet nn, Matrix inputs, Matrix results) {
    
    for (size_t l = 0; l < NN_COUNT(nn); ++l) {
        bzero(nn[l].w_grad, ROWS(nn[l].w_grad)*COLS((nn[l].w_grad))*sizeof(float));
        bzero(nn[l].b_grad, ROWS(nn[l].b_grad)*COLS((nn[l].b_grad))*sizeof(float));
    }

    for (size_t i = 0; i < ROWS(inputs); ++i) {
        nn_passthrough(nn, mat_at(inputs, i, 0));
        for (size_t j = 0; j < COLS(results); ++j) {
            *mat_at(NN_LAST_LAYER(nn).temp_grad, 0, j) = *mat_at(NN_LAST_LAYER(nn).result, 0, j) - *mat_at(results, i, j);
        }

        for (size_t l = NN_COUNT(nn); l-- > 0;) {
            if (l > 0) bzero(nn[l-1].temp_grad, ROWS(nn[l-1].temp_grad)*COLS(nn[l-1].temp_grad)*sizeof(float));
            for (size_t j = 0; j < COLS(nn[l].result); ++j) {
                float a = *mat_at(nn[l].result, 0, j);
                float da = *mat_at(nn[l].temp_grad, 0, j);
                *mat_at(nn[l].b_grad, 0, j) += 2*da*a*(1-a);
                for (size_t k = 0; k < COLS(nn[l].input); ++k) {
                    float pa = *mat_at(nn[l].input, 0, k);
                    float w = *mat_at(nn[l].w, k, j);
                    *mat_at(nn[l].w_grad, k, j) += 2*da*a*(1-a)*pa;
                    if (l > 0) *mat_at(nn[l-1].temp_grad, 0, k) += 2*da*a*(1-a)*w;
                }
            }
        }
    }

    for (size_t l = 0; l < NN_COUNT(nn); ++l) {
        for (size_t j = 0; j < ROWS(nn[l].w_grad); j++) {
            for (size_t k = 0; k < COLS(nn[l].w_grad); k++) {
                *mat_at(nn[l].w_grad, j, k) /= ROWS(inputs);
            }
        }

        for (size_t j = 0; j < ROWS(nn[l].b_grad); j++) {
            for (size_t k = 0; k < COLS(nn[l].b_grad); k++) {
                *mat_at(nn[l].b_grad, j, k) /= ROWS(inputs);
            }
        }
    }
}

void nn_learn(NeuralNet nn, float inputs[], float results[]) {
    for (size_t l = 0; l < NN_COUNT(nn); ++l) {
        for (size_t r = 0; r < ROWS(nn[l].w); ++r) {
            for (size_t c = 0; c < COLS(nn[l].w); ++c) {
                *mat_at(nn[l].w, r, c) -= RATE * *mat_at(nn[l].w_grad, r, c);
            }
        }

        for (size_t r = 0; r < ROWS(nn[l].b); ++r) {
            for (size_t c = 0; c < COLS(nn[l].b); ++c) {
                *mat_at(nn[l].b, r, c) -= RATE * *mat_at(nn[l].b_grad, r, c);
            }               
        }
    }
}

void nn_print(NeuralNet nn) {
    for (size_t l = 0; l < NN_COUNT(nn); ++l) {
        printf("===========\n");
        printf("============ Layer %ld Weights ============\n", l);
        mat_print(nn[l].w);
        printf("============ Layer %ld Biases ============\n", l);
        mat_print(nn[l].b);
        printf("===========\n");
    }
    printf("\n");
}

void nn_train(NeuralNet nn, Matrix inputs, Matrix results) {
    for (size_t i = 0; i < 100000; ++i) {
        // nn_finite_diff(nn, inputs, results);                
        nn_backprop(nn, inputs, results);
        nn_learn(nn, inputs, results);
        printf("cost - %lf\n", nn_cost(nn, inputs, results));
    }
}

int main(int argc, char **argv) {
    float inputs_arr[] = {
        1.0, 1.0,
        1.0, 0.0,
        0.0, 1.0,
        0.0, 0.0
    };

    Matrix inputs = mat_init(inputs_arr, 4, 2);

    float results_arr[] = {
        0.0,
        1.0,
        1.0,
        0.0
    };
    Matrix results = mat_init(results_arr, 4, 1);
    
    srandom(0x1337);
    size_t shapes[] = {2, 2, 1};
    NeuralNet nn = nn_init(2, 3, shapes);
    nn_train(nn, inputs, results);
    for (size_t r = 0; r < ROWS(inputs); ++r) {
        nn_passthrough(nn, mat_at(inputs, r, 0));
        printf("%lf ^ %lf = %lf\n", *mat_at(inputs, r, 0), *mat_at(inputs, r, 1), *mat_at(NN_LAST_LAYER(nn).result, 0, 0));
    }
} 
