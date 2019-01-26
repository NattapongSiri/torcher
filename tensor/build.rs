use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn implement_type_conversion_for_trait() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("conv.rs");
    let mut f = File::create(&dest_path).unwrap();
    let supported_types = [
        ("ByteTensor", "ByteStorage", "u8"), 
        ("CharTensor", "CharStorage", "i8"),
        ("FloatTensor", "FloatStorage", "f32"),
        ("DoubleTensor", "DoubleStorage", "f64"),
        ("IntTensor", "IntStorage", "i32"),
        ("LongTensor", "LongStorage", "i64"),
        ("ShortTensor", "ShortStorage", "i16")
    ];

    for src in &supported_types {
        for dest in &supported_types {
            if src != dest {
                f.write_all(format!("
                impl<'a, 'b: 'a> From<&'a {src}<'b>> for {dest}<'b> 
                {{
                    /// Perform type casting from {src} to {dest}.
                    /// This is done by deep cloning on entire storage while also
                    /// casting each element in storage from `u8` to `i8` at the same time.
                    /// The return tensor is completely independent from original tensor.
                    fn from(src: &'a {src}<'b>) -> {dest}<'b> {{
                        let (size, stride) = src.shape();
                        let src_data = src.data();
                        let mut storage = {dest_storage}::new_with_size(src_data.len());
                        let data = storage.data_mut();

                        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)| *dest = *src as {dest_ty});
                        {dest}::new_with_storage_nd(storage, 0, size, stride)
                    }}
                }}
                ",
                src=src.0,
                dest=dest.0,
                dest_storage=dest.1,
                dest_ty=dest.2
                ).as_bytes()).expect("Auto implementation error on implementing From for each tensor");
            }
        }
    }
}

fn main() {
    // implement auto conversion
    implement_type_conversion_for_trait();

    println!(r"cargo:rustc-link-search=clib");
}