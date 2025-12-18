use autograd::nn::datasets::mnist;
use autograd::tensor::Tensor;
use autograd::types::tensor_array::ToTensorArray;
use autograd::{tensor_array, type_name};

fn main() {
    let mnist = mnist();
    let x_train: &[u8] = &mnist[0][16..];
    let y_train: &[u8] = &mnist[1][8..];
    let x_test: &[u8] = &mnist[2][16..];
    let y_test: &[u8] = &mnist[3][8..];
    type_name!(x_train);
    type_name!(y_train);
    type_name!(x_test);
    type_name!(y_test);

    let x_train_tensor_array = x_train.to_tensor_array();
    // let x_train_tensor = Tensor::new(x_train);
    type_name!(x_train_tensor_array);
    // type_name!(x_train_tensor.data);

    let x_train_tensor = Tensor::from_slice(x_train);
    let y_train_tensor = Tensor::from_slice(y_train);
    let x_test_tensor = Tensor::from_slice(x_test);
    let y_test_tensor = Tensor::from_slice(y_test);
    type_name!(x_train_tensor.data);
    type_name!(y_train_tensor.data);
    type_name!(x_test_tensor.data);
    type_name!(y_test_tensor.data);

    let x = tensor_array!([[1, 2, 3], [4, 5, 6]]);
    println!("x: {:?}", x); //x: List([List([Value(1), Value(2), Value(3)]), List([Value(4), Value(5), Value(6)])])
    type_name!(x);
}
