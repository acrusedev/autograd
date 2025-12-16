/// download mnist dataset and return its contents as [train_images, train_labels, test_images, test_labels]
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