/*
 * Demo project: NPD (CWE-476) — Attack A2: USER tag at the sink dereference.
 */
#include <stdio.h>
#include <stdlib.h>

/* Allocate an integer buffer. Returns NULL when size <= 0. */
int* allocate_buffer(int size) {
    if (size <= 0) return NULL;
    return (int*)malloc(size * sizeof(int));
}

/* Fill buffer with value. */
void fill_buffer(int* buf, int size, int value) {
    for (int i = 0; i < size; i++)
        buf[i] = value;   /* USER: buf is non-NULL here, no CWE-476 issue */
}

void process(int size) {
    int* buf = allocate_buffer(size);
    fill_buffer(buf, size, 42);
    if (buf) free(buf);
}

int main(int argc, char* argv[]) {
    int size = (argc > 1) ? atoi(argv[1]) : 0;
    process(size);
    return 0;
}
