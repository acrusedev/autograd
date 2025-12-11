use tempdir::TempDir; // save mnist files to temp directory

// https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
// https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
// https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
// https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz

fn main() {
    download_mnist();
}

fn download_mnist() {
    let mnist_urls: Vec<&str> = vec![
        "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz",
        "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz",
        "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz",
        "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz",
    ];

    for url in mnist_urls {
        let resp = reqwest::blocking::get(url).unwrap();
        assert_eq!(resp.status().as_u16(), 200);
    }
}
