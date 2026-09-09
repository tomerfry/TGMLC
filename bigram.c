#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

// gcc -g bigram.c -lm

typedef uint8_t tok_t;

enum {TOK_START=0, TOK_END, TOK_FIRST_CHR};

typedef struct { tok_t a, b;} bigram_t;


char *tok_str(tok_t *t) {
    static char buf[2];
    if (*t == TOK_START) return "<S>";
    if (*t == TOK_END) return "<E>";
    buf[0] = *t-TOK_FIRST_CHR; buf[1] = 0;
    return buf;
}


size_t tokenize(char *str, size_t len, tok_t *out) {
    size_t n = 0;
    out[n++] = TOK_START;
    for (size_t i = 0; i < len; ++i)
        out[n++] = TOK_FIRST_CHR + str[i];
    out[n++] = TOK_END;
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


int main(int argc, char **argv) {
    char *data = "tomer\nkeren";
    size_t lines_count = 1;
    char *mut_data = strdup(data);
    char **lines = mut_split(mut_data, strlen(data), &lines_count);

    for (size_t i = 0; i < lines_count; ++i) {
        printf("%s\n", lines[i]);
        tok_t *t = (tok_t *)malloc(strlen(lines[i])+2);
        tokenize(lines[i], strlen(lines[i]), t);
        print_toks(t, strlen(lines[i])+2);
    }
} 

