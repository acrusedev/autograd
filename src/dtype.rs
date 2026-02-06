use std::fmt::Formatter;

#[derive(Debug)]
pub enum DType {
    float32,
    float64,
    unsigned8,
    signed8,
}

impl DType {
    pub fn get_bit_size(&self) -> isize {
        match self {
            DType::unsigned8 => 1,
            DType::signed8 => 1,
            DType::float32 => 4,
            DType::float64 => 8,
        }
    }

    pub fn format_char(&self) -> char {
        // https://numpy.org/devdocs/reference/arrays.dtypes.html
        match self {
            DType::unsigned8 => 'B',
            DType::signed8 => 'b',
            DType::float32 => 'f',
            DType::float64 => 'd',
        }
    }

    pub fn from_str(string: &str) -> DType {
        match string {
            "b" => DType::signed8,
            "B" => DType::unsigned8,
            "f" => DType::float32,
            "d" => DType::float64,
            _ => panic!("DType {} is not supported.", string),
        }
    }
}
