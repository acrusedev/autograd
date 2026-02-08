use std::fmt::Formatter;

#[derive(Debug)]
pub enum DType {
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
}

impl DType {
    pub fn get_bit_size(&self) -> isize {
        match self {
            DType::Bool => 1,
            DType::Int8 => 1,
            DType::Int16 => 2,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Uint8 => 1,
            DType::Float16 => 2,
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Bfloat16 => 2,
        }
    }

    pub fn format_char(&self) -> char {
        // https://numpy.org/devdocs/reference/arrays.dtypes.html
        match self {
            DType::Bool => '?',
            DType::Int8 => 'b',
            DType::Int16 => 'h',
            DType::Int32 => 'i',
            DType::Int64 => 'q',
            DType::Uint8 => 'B',
            DType::Float16 => 'e',
            DType::Float32 => 'f',
            DType::Float64 => 'd',
            DType::Bfloat16 => 'v',
        }
    }

    pub fn from_str(string: &str) -> DType {
        match string {
            "?" => DType::Bool,
            "b" => DType::Int8,
            "h" => DType::Int16,
            "i" => DType::Int32,
            "q" => DType::Int64,
            "B" => DType::Uint8,
            "e" => DType::Float16,
            "f" => DType::Float32,
            "d" => DType::Float64,
            "v" => DType::Bfloat16,
            _ => panic!("DType {} is not supported.", string),
        }
    }
}
