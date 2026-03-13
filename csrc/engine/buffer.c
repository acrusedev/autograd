#include "buffer.h"
#include "dtype.h"
#include "storage.h"
#include <stdlib.h>
#include <string.h>

Buffer *init_buffer(uint8_t* bytes, int bytes_len, size_t* shape, size_t* strides, int ndim, char fmt){
  Buffer* b = malloc(sizeof(Buffer));
  b->shape = malloc(sizeof(size_t) * ndim);
  memcpy(b->shape, shape, sizeof(size_t) * ndim);
  b->strides= malloc(sizeof(size_t) * ndim);
  memcpy(b->strides, strides, sizeof(size_t) * ndim);
  b->data = create_storage_from_slice(bytes, bytes_len);
  b->ndim = ndim;
  return b;
};

