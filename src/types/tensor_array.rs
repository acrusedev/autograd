#[derive(Debug)]
pub enum TensorArray<T> {
    Value(T),
    List(Vec<TensorArray<T>>),
}

/// Create a tensor array from an iterable.
#[macro_export]
macro_rules! tensor_array {
    ([]) => { $crate::types::tensor_array::TensorArray::List(Vec::new()) };
    ([$($inner:tt),* $(,)?]) =>
    {
        $crate::types::tensor_array::TensorArray::List(vec![$(tensor_array!($inner)),*])
    };
    (vec![$($inner:tt),* $(,)?]) => {
        $crate::types::tensor_array::TensorArray::List(vec![$(tensor_array!($inner)),*])
    };

    ($val: expr) => { $crate::types::tensor_array::TensorArray::Value($val) };
}

impl<T> TensorArray<T> {
    pub fn len(&self) -> usize {
        match self {
            TensorArray::Value(_) => 1,
            TensorArray::List(inner) => inner.len(),
        }
    }
}

/// Trait to convert iterables into tensor arrays.
pub trait ToTensorArray<T: Clone> {
    fn to_tensor_array(&self) -> Vec<T>;
}

/// Implementation of ToTensorArray for slices.
impl<T: Clone> ToTensorArray<T> for &[T] {
    fn to_tensor_array(&self) -> Vec<T> {
        self.to_vec()
    }
}

/// Trait to infer the shape of an iterable.
pub trait InferShape<T> {
    fn infer_shape(&self) -> Vec<usize>;
}

impl<T> InferShape<T> for &[T] {
    fn infer_shape(&self) -> Vec<usize> {
        println!("self len: {:?}", self.len());
        vec![self.len(), 1]
    }
}
