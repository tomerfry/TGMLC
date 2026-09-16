#include <nn.h>
#include <string.h>


int main(int argc, char **argv) {
    if (argc != 3) return -1;
    char *str = argv[1], *delim = argv[2];
    char * tok = strtok(str, delim);

    while (tok) {
        printf("%s\n", tok);
        tok = strtok(0, delim);
    }
    return 0;
} 

