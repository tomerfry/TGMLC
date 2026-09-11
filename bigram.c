#include <math.h>
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdint.h>

// gcc -g bigram.c -lm

typedef uint32_t tok_t;

enum {TOK_SEP=0, TOK_FIRST_CHR};

typedef struct { tok_t a, b;} bigram_t;

typedef struct {
    size_t ndims;
    size_t *shape;
} TensorHeader;

typedef uint32_t *Tensor;
typedef float *FloatTensor;

#define TENSOR_HEAD(t) ((TensorHeader *)(t) - 1)
#define TENSOR_SHAPE(t) (TENSOR_HEAD((t))->shape)


char *tok_str(tok_t *t) {
    static char buf[2];
    if (*t == TOK_SEP) return ".";
    buf[0] = *t-TOK_FIRST_CHR; buf[1] = 0;
    return buf;
}


size_t tokenize(char *str, size_t len, tok_t *out) {
    size_t n = 0;
    out[n++] = TOK_SEP;
    for (size_t i = 0; i < len; ++i)
        out[n++] = TOK_FIRST_CHR + str[i];
    out[n++] = TOK_SEP;
    return n;
};


void print_toks(tok_t *tokens, size_t len) {
    for (int i = 0; i < len; ++i) {
        printf("%s\n", tok_str(&tokens[i]));
    }
}


size_t count_chr(char *data, size_t len, char chr) {
    size_t n = 0;
    for (size_t i = 0; i < len; ++i) {
        if (data[i] == chr) n++;
    }
    return n;
}


char ** mut_split(char *data, size_t len, size_t *ptrs_count) {
    size_t n = 0;
    *ptrs_count = count_chr(data, len, '\n') + 1;
    char **out = malloc(*ptrs_count * sizeof(char *));
    memset(out, 0, *ptrs_count*sizeof(char *));

    out[0] = data;

    for (size_t i = 0; i < len; ++i) {
        if (data[i] == '\n') {
            data[i] = '\x00';
            out[++n] = &data[i+1];
        }
    }
    return out;
}


char *slice(char *data, size_t len, size_t start, size_t end) {
    if (start >= end) return NULL;
    if (end - start > len) return NULL;
    
    char *out = malloc(end - start);
    memcpy(out, &data[start], end-start);
    return out;
}


Tensor tensor_zeros(size_t ndims, size_t *shape) {
    size_t prod = 0;
    for (int i = 0; i < ndims; ++i) {
        prod *= shape[i];
    }
    TensorHeader *t_head = malloc(ndims*sizeof(size_t)+prod*sizeof(uint32_t));
    t_head->ndims = ndims;
    t_head->shape = malloc(ndims*sizeof(size_t));
    memcpy(t_head->shape, shape, ndims*sizeof(size_t));
    bzero((Tensor)(t_head+1), prod*sizeof(uint32_t));
    return (Tensor)(t_head + 1);   
}


Tensor tensor_init(uint32_t *t, size_t ndims, size_t *shape) {
    size_t prod = 0;
    for (int i = 0; i < ndims; ++i) {
        prod *= shape[i];
    }
    TensorHeader *t_head = malloc(ndims*sizeof(size_t)+prod*sizeof(uint32_t));
    t_head->ndims = ndims;
    memcpy(&(t_head->shape), shape, ndims*sizeof(size_t));
    memcpy((Tensor)(t_head+1), t, prod*sizeof(uint32_t));
    return (Tensor)(t_head + 1);   
}


Tensor tensor_at(Tensor t, size_t coord[]) {
    TENSOR_HEAD(t)->shape;
    size_t offset = 0;
    size_t dim = 0;
    for (;dim++ < TENSOR_HEAD(t)->ndims-1;) {
        offset += coord[dim] * TENSOR_HEAD(t)->shape[dim+1];
    }
    offset += coord[dim];
    return &t[offset];
}

char *get_vocab(char *data, size_t len) {
    char *vocab = (char *)malloc(len);
    size_t placed = 0;
    bzero(vocab, len);

    for (size_t i = 0; i < len; ++i) {
        size_t j = 0;
        for (; j < placed; ++j) {
            if (vocab[j] == data[i]) break; 
        }
        if (j == placed || placed == 0) {
            if (data[i] != '\n') {
                vocab[placed] = data[i];
                placed++;
            }
        }

    }
    return vocab; 
}


void tensor_print(Tensor t) {
    if (TENSOR_HEAD(t)->ndims != 2) return;

    size_t coord[] = {0, 0};
    for (size_t x = 0; x < TENSOR_HEAD(t)->shape[0]; ++x) {
        for (size_t y = 0; y < TENSOR_HEAD(t)->shape[1]; ++y) {
            coord[0] = x;
            coord[1] = y;
            printf("%d ", *tensor_at(t, coord));
        }
        printf("\n");
    }
}


int main(int argc, char **argv) {
    char *raw = "tomer\nkeren";
    char *lines[] = {
        "tomer",
        "keren"
    };
    size_t lines_count = 2;
    
    char * vocab = get_vocab(raw, strlen(raw));
    size_t shape[] = {0, 0};
    shape[0] = strlen(vocab) + 1;
    shape[1] = strlen(vocab) + 1;
    Tensor bigram = tensor_zeros(2, shape);
    tensor_print(bigram);
    
    for (size_t i = 0; i < lines_count; ++i) {
        tok_t *t = (tok_t *)malloc(strlen(lines[i])+2);
        tokenize(lines[i], strlen(lines[i]), t);
        print_toks(t, strlen(lines[i])+2);
        free(t);
    }
} 

