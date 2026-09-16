#include <nn.h>

int main(int argc, char **argv) {

    float inputs_arr[N_SAMPLES * 2*BITS];      // 8 per row
    float results_arr[N_SAMPLES * (BITS+1)];   // 5 per row

    for (size_t a = 0; a < (1 << BITS); ++a) {
        for (size_t b = 0; b < (1 << BITS); ++b) {
            size_t row = a * (1 << BITS) + b;
            size_t sum = a + b;

            for (size_t bit = 0; bit < BITS; ++bit) {
                inputs_arr[row*2*BITS + bit]        = (a >> bit) & 1;
                inputs_arr[row*2*BITS + BITS + bit] = (b >> bit) & 1;
            }
            for (size_t bit = 0; bit < BITS+1; ++bit) {
                results_arr[row*(BITS+1) + bit] = (sum >> bit) & 1;
            }
        }
    }

    Matrix inputs  = mat_init(inputs_arr,  N_SAMPLES, 2*BITS);
    Matrix results = mat_init(results_arr, N_SAMPLES, BITS+1);

    srandom(0x1337);
    size_t shapes[] = {8, 16, 5};
    NeuralNet nn = nn_init(8, 3, shapes);
    nn_train(nn, inputs, results);

    size_t correct = 0;
    for (size_t a = 0; a < (1 << BITS); ++a) {
        for (size_t b = 0; b < (1 << BITS); ++b) {
            size_t row = a * (1 << BITS) + b;
            nn_passthrough(nn, mat_at(inputs, row, 0));

            size_t pred = 0;
            for (size_t bit = 0; bit < BITS+1; ++bit)
                if (*mat_at(NN_LAST_LAYER(nn).result, 0, bit) > 0.5f) pred |= 1 << bit;

            int ok = (pred == a + b);
            correct += ok;

            printf("%s%2zu + %2zu = %2zu", ok ? "  " : "! ", a, b, pred);
            if (!ok) printf("  (want %2zu)", a + b);

            printf("   [");
            for (size_t bit = BITS; bit + 1 > 0; --bit)          // MSB first
                printf("%.2f%s", *mat_at(NN_LAST_LAYER(nn).result, 0, bit), bit ? " " : "");
            printf("]\n");
        }
    }
    printf("\n%zu / %d correct\n", correct, N_SAMPLES);
} 

