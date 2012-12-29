#ifndef LIB_H
#define LIB_H

typedef struct {
    int x;
    int y;
    int z;
} dim3;

extern void write_uint(int fd, unsigned int val);
extern void write_str(int fd, const char *str);
extern void write_pair(int fd, const char *str, unsigned int val);

extern dim3 threadIdx;
extern dim3 threadDim;
extern dim3 blockIdx;
extern dim3 blockDim;

#endif /* LIB_H */
