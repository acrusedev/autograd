pub fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>());
}

#[macro_export]
macro_rules! type_name {
    ($expr:expr) => {
        $crate::helpers::print_type_of(&$expr);
    };
}
