#include <nn.h>
#include <malloc.h>
#include <string.h>

#define EMB_SZ (2)
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
    size_t i = c - 'a';
    if (i < EMB_COUNT) return i;
}


char itoc(size_t i) {
    if (i == 0) return '.';
    if (i < EMB_COUNT) return i + 'a';
}


Matrix ctom(char c) {
    Matrix one_hot = mat_zeros(1, EMB_COUNT);
    if (c == '.') *mat_at(one_hot, 0, 0) = 1.0f;
    else *mat_at(one_hot, 0, ctoi(c)) = 1.0f;
}


char mtoc(Matrix one_hot) {
    float biggest = *mat_at(one_hot, 0, 0);
    size_t idx = 0;
    for (size_t i = 0; i < EMB_COUNT; ++i) {
        if (*mat_at(one_hot, 0, i) > biggest) {
            biggest = *mat_at(one_hot, 0, i);
            idx = i;
        }
        i++;
    }
    
    return itoc(idx);
}


Matrix compile_dataset(char *str, size_t items_count, Matrix emb) {
    char **items = (char **)malloc(items_count * sizeof(char *));
    bzero(items, items_count * sizeof(char *));

    char * tok = strtok(str, "\n");
    size_t idx = 0;

    while (tok) {
        items[idx++] = tok;
        tok = strtok(0, "\n");
    }

    Matrix dataset = mat_zeros(strlen(str), CONTEXT_SZ*EMB_SZ);

    size_t row = 0;
    char context[CONTEXT_SZ+1] = { 0 };
    for (size_t i = 0; i < items_count; ++i) {
        char * word = items[i];
        printf("%s\n", word);
        for (int offset = 1-CONTEXT_SZ; offset < (int)strlen(word)-CONTEXT_SZ+1; ++offset) {
            int left = snprintf(context, CONTEXT_SZ+1, "%.*s", offset < 0 ? offset * -1 : 0, "...");
            (void)snprintf(context+left, CONTEXT_SZ-left+1, "%.*s", CONTEXT_SZ-left, &word[offset < 0 ? 0 : offset]);
            printf("%s --> %c\n", context, word[offset + CONTEXT_SZ]);
        }
        printf("\n");
    }
}


int main(int argc, char **argv) {
    char *str = strdup("abcdefghijklmnop\nabcdefghijklmnop");

    srand(0x1337);
    Matrix embeddings = mat_rand(EMB_COUNT, EMB_SZ); 

    Matrix dataset = compile_dataset(str, 2, embeddings);
    return 0;
}

