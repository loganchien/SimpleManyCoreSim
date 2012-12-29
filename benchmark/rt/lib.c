#include "lib.h"

#include <unistd.h>
#include <string.h>

dim3 threadIdx = { 0, 0, 0 };
dim3 threadDim = { 0, 0, 0 };
dim3 blockIdx = { 0, 0, 0 };
dim3 blockDim = { 0, 0, 0 };

void write_uint(int fd, unsigned int val) {
    int i;
    static const char hex[] = "0123456789abcdef";
    char buf[64];
    char *ptr = buf + sizeof(buf) - 1;
    for (i = 0; i < sizeof(unsigned int) * 2; ++i) {
        *ptr-- = hex[val & 0xf];
        val >>= 4;
    }
    ++ptr;
    write(fd, ptr, buf + sizeof(buf) - ptr);
}

void write_str(int fd, const char *str) {
    size_t len = strlen(str);
    write(fd, str, len);
}

void write_pair(int fd, const char *str, unsigned int val) {
    write_str(fd, str);
    write_uint(fd, val);
    write(fd, "\n", 1);
}
