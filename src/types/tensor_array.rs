#[derive(Debug)]
pub enum TensorArray<T> {
    Value(T),
    List(Vec<TensorArray<T>>),
}

#[macro_export]
macro_rules! tensor_array {
    ([]) => { $create::types::tensor_array::TensorArray::List(Vec::new()) };
    ($val: expr) => { $crate::types::tensor_array::TensorArray::Value($val) };
    ([$($inner:tt),* $(,)?]) => {
        $create::types::tensor_array::TensorArray::List(vec![$(tensor_array!($inner)),*])
    };
}
