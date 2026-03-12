#ifndef STORAGE_H
#define STORAGE_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
  size_t len;
  uint8_t data[];
} Storage;

Storage *create_storage_from_slice(uint8_t *slice, size_t len);
void storage_free(Storage *s);
#endif
