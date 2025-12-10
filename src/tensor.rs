struct Tensor {
    data: Vec<f64>,
}

impl Tensor {
    fn new(data: Vec<f64>) -> Self {
        Self { data }
    }
}
