#include "storage.h"
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

Storage *create_storage_from_slice(uint8_t *slice, size_t len) {
  Storage*s = malloc(sizeof(Storage) + len*sizeof(uint8_t));
  s->len = len;
  memcpy(s->data, slice, len);
  return s;
}

void storage_free(Storage *s) {
  free(s);
}
