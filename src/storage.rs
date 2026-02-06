pub struct Storage {
    data: Vec<u8>, // single memory address stores 8bits of data
}

impl Storage {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
    pub fn from_slice(s: &[u8]) -> Storage {
        Storage { data: s.to_owned() }
    }
}
