#include "lib.h"

#include <unistd.h>
#include <string.h>

dim2 threadIdx = { 0, 0 };
dim2 threadDim = { 0, 0 };
dim2 blockIdx = { 0, 0 };
dim2 blockDim = { 0, 0 };

void utoa_hex(char *output, unsigned int val) {
    static const char hex[] = "0123456789abcdef";

    unsigned int i;
    unsigned int shift = 28u;

    for (i = 0; i < 8; ++i, shift -= 4u) {
        *output++ = hex[(val >> shift) & 0xfu];
    }
}

void write_uint(int fd, unsigned int val) {
    char buf[16];
    utoa_hex(buf, val);
    write(fd, buf, 8);
}

void write_str(int fd, const char *str) {
    size_t len = strlen(str);
    write(fd, str, len);
}

void write_pair(int fd, const char *str, unsigned int val) {
    char buf[1024];
    char *buf_ptr = buf;
    unsigned buf_len = sizeof(buf) - 1;
    unsigned str_len = strlen(str);

    if (str_len < buf_len) {
        memcpy(buf_ptr, str, str_len + 1);
        buf_len -= str_len;
        buf_ptr += str_len;
    }

    if (buf_len > 8) {
        utoa_hex(buf_ptr, val);
        buf_len -= 8;
        buf_ptr += 8;
        *buf_ptr = '\0';
    }

    if (buf_len > 1) {
        buf_len -= 1;
        *buf_ptr++ = '\n';
        *buf_ptr = '\0';
    }

    write(fd, buf, buf_ptr - buf);
}

void write_thread_info(int fd) {
    char buf[] = "[threadIdx: XXXXXXXX XXXXXXXX , blockIdx: XXXXXXXX XXXXXXXX]\n";

    utoa_hex(buf + 12, threadIdx.y);
    utoa_hex(buf + 21, threadIdx.x);
    utoa_hex(buf + 42, blockIdx.y);
    utoa_hex(buf + 51, blockIdx.x);

    write(fd, buf, sizeof(buf) - 1);
}
