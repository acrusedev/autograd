struct TensorArray {
    // Represents a multi-dimensional, homogenous array of fixed items.
    shape: Vec<usize>,
}

impl TensorArray {
    fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }

    fn zeros(shape: Vec<usize>) -> Self {
        Self { shape }
    }

    fn empty(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}
