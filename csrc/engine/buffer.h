#ifndef BUFFER_H
#define BUFFER_H
#include "dtype.h"
#include "storage.h"

typedef struct {
  int ndim;
  size_t *shape;
  size_t *strides;
  DType dtype;
  Storage* data;
} Buffer;

Buffer *init_buffer(uint8_t* bytes, int bytes_len, size_t* shape, size_t* strides, int ndim, char fmt);
#endif
