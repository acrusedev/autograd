pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T: Clone> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self { data, shape: shape }
    }

    pub fn from_slice(slice: &[T]) -> Self {
        Self {
            data: slice.to_vec(),
            shape: vec![slice.len(), 1],
        }
    }

    /// Reshape the tensor into a new shape.
    /// The original data must be compatible with the new shape.
    /// This changes tensor in place -> does not return a new tensor.
    pub fn reshape(&self, shape: Vec<usize>) {}
}
