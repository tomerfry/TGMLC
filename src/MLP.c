#include <nn.h>
#include <malloc.h>
#include <string.h>

#define EMB_SZ (10)
#define EMB_COUNT (28)
#define CONTEXT_SZ (3)


void split(char *str, char *delim) {
    char * tok = strtok(str, delim);

    while (tok) {
        printf("%s\n", tok);
        tok = strtok(0, delim);
    }
} 


size_t ctoi(char c) {
    if (c == '.') return 0;
    size_t i = c - 'a' + 1;
    if (i < EMB_COUNT) return i;
}


char itoc(size_t i) {
    if (i == 0) return '.';
    if (i < EMB_COUNT) return i + 'a' - 1;
}


void ctoemb(char c, Matrix emb, Matrix res) {

    if (c == '.') {
        memcpy(res, mat_at(emb, 0, 0), EMB_SZ*sizeof(float));
    } else {
       memcpy(res, mat_at(emb, c - 'a' + 1, 0), EMB_SZ*sizeof(float)); 
    }
}


void embtoc(Matrix emb, Matrix src, char *c) {
    size_t i = 0;
    for (; i < ROWS(emb); ++i) {
        size_t offset = 0;
        for (size_t j = 0; j < COLS(emb); ++j) {
            if (*mat_at(emb, i, j) != *mat_at(src, 0, offset)) break;
            else offset++;
        }
        if (offset == COLS(emb)) break;
    }
    if (i < ROWS(emb)) i == 0 ? (*c = '.') : (*c = i + 'a' - 1);
}



void compile_dataset(char *str, size_t items_count, Matrix emb, Matrix input, Matrix output) {
    char **items = (char **)malloc(items_count * sizeof(char *));
    bzero(items, items_count * sizeof(char *));

    char * tok = strtok(str, "\n");
    size_t idx = 0;

    while (tok) {
        items[idx++] = tok;
        tok = strtok(0, "\n");
    }


    size_t row = 0;
    char context[CONTEXT_SZ+1] = { 0 };
    for (size_t i = 0; i < items_count; ++i) {
        char * word = items[i];
        for (int offset = -CONTEXT_SZ; offset < (int)strlen(word)-CONTEXT_SZ; ++offset) {
            int left = snprintf(context, CONTEXT_SZ+1, "%.*s", offset < 0 ? offset * -1 : 0, "...");
            (void)snprintf(context+left, CONTEXT_SZ-left+1, "%.*s", CONTEXT_SZ-left, &word[offset < 0 ? 0 : offset]);
            printf("%s --> %c\n", context, offset + CONTEXT_SZ < (int)strlen(word) ? word[offset + CONTEXT_SZ] : '.');
            
            for (size_t i = 0; i < CONTEXT_SZ; ++i) {
                ctoemb(context[i], emb, mat_at(input, row, i*EMB_SZ));
            }
            ctoemb(offset + CONTEXT_SZ < (int)strlen(word) ? word[offset + CONTEXT_SZ] : '.', emb, mat_at(output, row, 0));
            row++;
        }
    }
}


int main(int argc, char **argv) {
    char *str = strdup("abcdefghijklmnop\nabcdefghijklmnop");

    Matrix embeddings = mat_rand(EMB_COUNT, EMB_SZ); 
    Matrix input = mat_zeros(strlen(str)-1, CONTEXT_SZ*EMB_SZ);
    Matrix output = mat_zeros(strlen(str)-1, EMB_SZ);
    compile_dataset(str, 2, embeddings, input, output);

    size_t shapes[] = {EMB_SZ*CONTEXT_SZ, 50, 28};
    NeuralNet nn = nn_init(EMB_SZ*CONTEXT_SZ, 3, shapes);
    nn_train(nn, input, output);

    return 0;
}

