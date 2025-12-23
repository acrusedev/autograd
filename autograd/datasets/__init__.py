def mnist(cache=False):
    import requests
    base_url = "https://raw.githubusercontent.com/fgnt/mnist/master/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    contents = []

    for file in files:
        response = requests.get(base_url + file)
        response.raise_for_status()
        if cache:
            import hashlib
            with open(file, "wb") as f:
                f.write(response.content)
        contents.append(response.content)

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