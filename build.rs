use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn implement_tensor_construction() {
    fn implement_macro(supported_types: &[(&str, &str)], f: &mut File) {
        f.write_all(b"
        /// Populate either known size tensor or populate a
        /// tensor then derive the size based on populated Vec
        #[macro_export]
        macro_rules! populate_tensor {").expect("Fail to auto implement populate_tensor macro header");
        for tar in supported_types {
            f.write_all(format!("
                ({ty}, $size: expr, $cb: expr) => (
                    $crate::generate_sized_{ty}_tensor($size, $cb)
                );
                ({ty}, $generator: expr) => (
                    {{
                        $crate::generate_unsized_{ty}_tensor($generator)
                    }}
                );
            ", 
            ty=tar.1
            ).as_bytes()).expect("Fail to auto implement populate_tensor code macro");
        }
        f.write_all(b"}").expect("Fail to auto implement populate_tensor macro close brace");
    };

    fn implement_private_fn(supported_types: &[(&str, &str)], f: &mut File) {
        for tar in supported_types {
            f.write_all(format!("
            /// Perform tensor construction for data type {ty}. It'll create an empty 1d {ts}
            /// then populate it by call the callback function on each element of mutable data ref
            /// along with the current index of given data.
            pub fn generate_sized_{ty}_tensor(size: usize, cb: impl FnMut((usize, &mut {ty}))) -> {ts} {{
                let mut tensor = {ts}::new_with_size_1d(size);
                tensor.data_mut().iter_mut().enumerate().for_each(cb);
                tensor
            }}

            /// Perform data populate by using generator closure which expected to return a 
            /// slice of the {ty}. It then perform deep copy of each
            pub fn generate_unsized_{ty}_tensor<'a>(mut generator: impl FnMut() -> &'a [{ty}]) -> {ts} {{
                {ts}::from(generator())
            }}
            ",
            ts=tar.0,
            ty=tar.1,
            ).as_bytes()).expect("Auto implementation error on implementing tensor generating private functions");
        }
    };

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("pop_tensor.rs");
    let mut f = File::create(&dest_path).unwrap();
    let supported_types = [
        ("ByteTensor", "u8"), 
        ("CharTensor", "i8"),
        ("FloatTensor", "f32"),
        ("DoubleTensor", "f64"),
        ("IntTensor", "i32"),
        ("LongTensor", "i64"),
        ("ShortTensor", "i16")
    ];

    implement_macro(&supported_types, &mut f);
    implement_private_fn(&supported_types, &mut f);
}

fn main() {
    implement_tensor_construction();

    println!(r"cargo:rustc-link-search=clib");
}