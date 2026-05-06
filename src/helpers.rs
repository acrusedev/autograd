pub fn calc_strides(shape: &[isize], itemsize: isize) -> Vec<isize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![itemsize; shape.len()];
    for n in (0..shape.len() - 2).rev() {
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
