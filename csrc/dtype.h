#ifndef DTYPE_H
#define DTYPE_H
#include <stddef.h>
typedef enum {
    Bool,
    // integer data types
    Int8,
    Int16,
    Int32,
    Int64,
    // unsigned integer
    Uint8,
    // floating point datat types
    Float16,
    Float32,
    Float64,
    Bfloat16,
} DType;

size_t get_byte_size(DType dtype);
char format_char(DType dtype);
DType dtype_from_str(char string);
#endif
