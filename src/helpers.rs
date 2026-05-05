use crate::dtype::DType;

pub fn calc_strides(shape: &[isize], dtype: &DType) -> Vec<isize> {
   match dtype {
       DType::Int32 => {
           vec![1]
       },
       _ => {
           vec![1]
       }
   }
}