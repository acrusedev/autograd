use pyo3::PyRef;

use crate::buffer::Buffer;

pub fn calc_strides(shape: &[isize], itemsize: isize) -> Vec<isize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![itemsize; shape.len()];
    for n in (0..shape.len() - 1).rev() {
        strides[n] = strides[n + 1] * shape[n + 1]
    }
    return strides;
}

pub fn get_coords(shape: &[isize], mut index: usize) -> Vec<isize> {
    let mut coords = vec![0isize; shape.len()];
    for i in (0..shape.len()).rev() {
        let dim = shape[i] as usize;
        coords[i] = (index % dim) as isize;
        index /= dim;
    }
    return coords;
}

pub fn numpy<T>(numel: usize, num_cols: usize, slice: &[T]) -> String {
    let v = slice.to_vec();
    let mut s = String::from("<Tensor [");
    for (index, element) in v.iter().enumerate() {
        if (index + 1) != numel {
            _ = write!(&mut s, "{}, ", element.to_string());
        } else {
            _ = write!(&mut s, "{}", element.to_string());
        }
        if (index + 1) % num_cols == 0 {
            s.push_str("\n\t");
        }
    }
    s.push_str(&format!("]>, dtype={}", tensor.dtype));
    s
}
