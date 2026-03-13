#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "dtype.h"
void* allocate_mem_cpu(size_t size) {
 return malloc(size);
}

void copy_in_cpu(uint8_t* src, uint8_t* dest, size_t len) {
    memcpy(dest, src, len);
}