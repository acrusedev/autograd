use std::{fs::File, io::Write};
use tempdir::TempDir; // save mnist files to temp directory

// https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
// https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
// https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
// https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz

fn main() {
    download_mnist();
}

fn download_mnist() {
    let mnist_urls = vec![
        "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz",
        "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz",
        "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz",
        "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz",
    ];

    let tmp_dir = TempDir::new("mnist").unwrap();
    println!("Temp dir: {:?}", tmp_dir.path());

    for url in mnist_urls {
        let resp = reqwest::blocking::get(url).unwrap();
        let file_path = tmp_dir.path().join(url.split('/').last().unwrap());
        let mut file = File::create(&file_path).unwrap();
        file.write_all(&resp.bytes().unwrap()).unwrap();
        println!("Saved: {:?}", file_path);
    }
}
