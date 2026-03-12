#include "dtype.h"
#include <assert.h>

size_t get_byte_size(DType dtype) {
  switch (dtype) {
    case Bool:
      return 1;
    case Int8:
      return 1;
    case Int16:
      return 2;
    case Int32:
      return 4;
    case Int64:
      return 8;
    case Uint8:
      return 1;
    case Float16:
      return 2;
    case Float32:
      return 4;
    case Float64:
      return 8;
    case Bfloat16:
      return 2;
  }
}

char format_char(DType dtype) {
  switch (dtype) {
    case Bool:
      return '?';
    case Int8:
      return 'b';
    case Int16:
      return 'h';
    case Int32:
      return 'i';
    case Int64:
      return 'q';
    case Uint8:
      return 'B';
    case Float16:
      return 'e';
    case Float32:
      return 'f';
    case Float64:
      return 'd';
    case Bfloat16:
      return 'v';
  }
}

DType dtype_from_str(char string) {
  switch (string) {
    case '?':
      return Bool;
    case 'b':
      return Int8;
    case 'h':
      return Int16;
    case 'i':
      return Int32;
    case 'q':
      return Int64;
    case 'B':
      return Uint8;
    case 'e':
      return Float16;
    case 'f':
      return Float32;
    case 'd':
      return Float64;
    case 'v':
      return Bfloat16;
    default:
      assert(0);
  }
}
