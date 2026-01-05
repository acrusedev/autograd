from autograd.helpers import fetch
from autograd.tensor import Tensor

def mnist():
    import requests
    base_url = "https://raw.githubusercontent.com/fgnt/mnist/master/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]


    def _mnist(file): return Tensor.from_url(base_url+file)

    contents = []

    for file in files:
        contents.append(_mnist(file))
        
    return contents


"""
pub fn mnist() -> Vec<Vec<u8>> {
    let base_url = String::from("https://raw.githubusercontent.com/fgnt/mnist/master/");
    let urls = vec![
        base_url.clone() + "train-images-idx3-ubyte.gz",
        base_url.clone() + "train-labels-idx1-ubyte.gz",
        base_url.clone() + "t10k-images-idx3-ubyte.gz",
        base_url.clone() + "t10k-labels-idx1-ubyte.gz",
    ];
    let mut arr_to_ret: Vec<Vec<u8>> = Vec::new();
    for url in urls {
        let resp = reqwest::blocking::get(url).unwrap();
        let bytes = resp.bytes().unwrap();
        arr_to_ret.push(bytes.to_vec());
    }
    arr_to_ret
}
"""
