#ifndef LIB_H
#define LIB_H

typedef struct {
    int y;
    int x;
} dim2;

extern void utoa_hex(char *output, unsigned int val);

extern void write_uint(int fd, unsigned int val);
extern void write_str(int fd, const char *str);
extern void write_pair(int fd, const char *str, unsigned int val);

extern void write_thread_info(int fd);

extern dim2 threadIdx;
extern dim2 threadDim;
extern dim2 blockIdx;
extern dim2 blockDim;

#endif /* LIB_H */
